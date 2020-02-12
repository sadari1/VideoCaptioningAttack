import json

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from multiprocessing import Pool
from multiprocessing import Queue


class CocoDataset(Dataset):

    def __init__(self, coco_labels):
        # python 3
        # super().__init__()
        super(CocoDataset, self).__init__()
        self.coco_labels = list(coco_labels['labels'].items())
        self.num_classes = coco_labels['num_classes']

    def __getitem__(self, ix):
        labels = torch.zeros(self.num_classes)
        image_id, labels_ids = self.coco_labels[ix]
        labels[labels_ids] = 1
        data = {}
        data['image_ids'] = image_id
        data['labels'] = labels
        return data

    def __len__(self):
        return len(self.coco_labels)


pool_queue = Queue()
work = []


def _threaded_sample_load(vid_id, fpath, n_frame_steps):
    fc_feat = load_and_subsample_feat(fpath, n_frame_steps)
    pool_queue.put((vid_id, fc_feat))


class VideoDataset(Dataset):

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt, mode):
        # python 3
        # super().__init__()
        super(VideoDataset, self).__init__()
        self.mode = mode  # to load train/val/test data

        # load the json file which contains information about the dataset
        # self.dataset_meta = json.load(open("D:\\College\\Research\\December 2018 Video Captioning Attack\\video captioner\\dataset_msvd.json"))
        self.dataset_meta = json.load(open(opt["dataset_json"]))
        self.vid_to_meta = self.dataset_meta['vid_to_meta']
        self.ix_to_word = self.dataset_meta['ix_to_word']
        self.word_to_ix = self.dataset_meta['word_to_ix']
        print('vocab size is ', len(self.ix_to_word))
        self.splits = self.dataset_meta['split_to_ids']
        print('number of train videos: ', len(self.splits['train']))
        print('number of val videos: ', len(self.splits['val']))
        print('number of test videos: ', len(self.splits['test']))
        self.n_frame_steps = opt['n_frame_steps']
        self.feats_dir = opt["feats_dir"][0]
        # self.c3d_feats_dir = opt['c3d_feats_dir']
        # self.with_c3d = opt['with_c3d']
        # load in the sequence data
        self.max_len = opt["max_len"]
        print('max sequence length in data is', self.max_len)

        if self.mode != 'inference':
            print('load feats from %s' % (self.feats_dir))
            # Memory cache for features
            print("Pre-cache {} features in memory.".format(len(self.splits[mode])))
            self._feat_cache = {}
            # pool = Pool(16)

            for fid in self.splits[mode]:
                fc_feat_path = os.path.join(self.feats_dir, fid)
                try:
                    fc_feat = load_and_subsample_feat(fc_feat_path, self.n_frame_steps)
                    self._feat_cache[fid] = fc_feat
                except:
                    print(fc_feat_path, "was not found")
                # work.append((fid, fc_feat_path, self.n_frame_steps))
            # pool.starmap(_threaded_sample_load, work)
            # pool.close()
            # pool.join()

            # while not pool_queue.empty():
            #     key, feat = pool_queue.get()
            #     self._feat_cache[key] = feat

        print("Finished initializing dataloader.")

    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = ix % len(self.splits[self.mode])
        vid_id = self.splits[self.mode][ix]

        # if self.mode == 'val':
        #     ix += len(self.splits['train'])
        # elif self.mode == 'test':
        #     ix = ix + len(self.splits['train']) + len(self.splits['val'])

        # for dir in self.feats_dir:
        #     fc_feat.append(np.load(os.path.join(dir, 'video%i.npy' % (ix))))

        if vid_id not in self._feat_cache.keys():
            fc_feat_path = os.path.join(self.feats_dir, vid_id)
            fc_feat = load_and_subsample_feat(fc_feat_path)
            self._feat_cache[vid_id] = fc_feat
        else:
            fc_feat = self._feat_cache[vid_id]

        # fc_feat = np.concatenate(fc_feat, axis=1)

        # if self.with_c3d == 1:
        #     c3d_feat = np.load(os.path.join(self.c3d_feats_dir, 'video%i.npy'%(ix)))
        #     if len(c3d_feat.shape) == 1:
        #         fc_feat = np.concatenate((fc_feat, np.tile(c3d_feat, (fc_feat.shape[0], 1))), axis=1)
        #     elif len(c3d_feat.shape) == 2:
        #         samples = np.round(np.linspace(
        #             0, c3d_feat.shape[0] - 1, fc_feat.shape[0])).astype(np.int32)
        #         fc_feat = np.concatenate((fc_feat, c3d_feat[samples, :]), axis=1)
        # label = torch.zeros(self.max_len)
        mask = torch.zeros(self.max_len)
        captions = self.vid_to_meta[vid_id]['final_captions']
        gts = torch.zeros(len(captions), self.max_len).long()
        for i, cap in enumerate(captions):
            if len(cap) > self.max_len:
                cap = cap[:self.max_len]
                cap[-1] = '<eos>'
            for j, w in enumerate(cap):
                gts[i, j] = self.word_to_ix[w]
                # # add by rgh
                # if w in self.word_to_ix.keys():
                #     gts[i, j] = self.word_to_ix[w]
                # else:
                #     gts[i, j] = 0

        # random select a caption for this video
        cap_ix = random.randint(0, len(captions) - 1)
        label = gts[cap_ix]
        non_zero = (label == 0).nonzero()
        mask[:int(non_zero[0]) + 1] = 1

        data = {}
        data['fc_feats'] = Variable(torch.from_numpy(fc_feat).type(torch.FloatTensor))
        # Just need to sample one gt for now. Eventually incorporate n captions (for fixed n)
        data['labels'] = Variable(label.long())
        data['masks'] = Variable(mask)
        # data['gts'] = gts
        data['video_ids'] = ix
        return data

    def __len__(self):
        return len(self.splits[self.mode])


def load_and_subsample_feat(fc_feat_path, n_frame_steps=28):
    fc_feat = np.load(fc_feat_path + '.npy')
    # Subsampling
    samples = np.round(np.linspace(
        0, fc_feat.shape[0] - 1, n_frame_steps)).astype(np.int32)
    try:
        fc_feat = fc_feat[samples, :]
    except Exception as e:
        print("Bad feature file in dataset: {}. Purge, re-process, and try again.".format(fc_feat_path))
        raise e
    return fc_feat
