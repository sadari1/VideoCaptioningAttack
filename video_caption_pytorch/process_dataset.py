"""
Process dataset meta files to something that can be turned into VideoDataset

For MSVD, we will rely on some files created by Li Yao in the temporal attention repo:
https://github.com/yaoli/arctic-capgen-vid
"""
import pickle
import os
import argparse
import re
import json

'''
--gtdict
"D:\College\Research\2019 Video Captioning Attack Conference Paper\youtube2text_preprocessed_for_arctic_capgen_vid\youtube2text_iccv15\dict_movieID_caption.pkl"
--out_json_basepath
"D:\College\Research\2019 Video Captioning Attack Conference Paper\vgg16_dataset_"

'''

def main(args):
    dict_path = args.gtdict
    count_thr = args.threshold
    dataset = args.dataset
    if not os.path.exists(dict_path):
        raise FileNotFoundError("File not found: {}".format(dict_path))

    counts = {}
    if dataset == 'msvd':
        gtdict, split_to_ids = process_msvd(dict_path)
    else:
        raise NotImplementedError("Dataset not implemented: {}".format(dataset))

    for vid, caps in gtdict.items():
        for cap in caps:
            cap_e = cap.decode('utf-8')
            ws = re.sub(r'[.!,;?]', ' ', cap_e).split()
            for w in ws:
                counts[w] = counts.get(w, 0) + 1

    total_words = sum(counts.values())
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' %
          (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' %
          (bad_count, total_words, bad_count * 100.0 / total_words))

    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('<UNK>')

    vid_to_meta = {}
    for vid, caps in gtdict.items():
        final_captions = []
        for cap in caps:
            cap_e = cap.decode('utf-8')
            ws = re.sub(r'[.!,;?]', ' ', cap_e).split()
            caption = [
                '<sos>'] + [w if counts.get(w, 0) > count_thr else '<UNK>' for w in ws] + ['<eos>']
            final_captions.append(caption)
        vid_to_meta[vid.decode('utf-8')] = {'final_captions': final_captions}

    itow = {i + 2: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 2 for i, w in enumerate(vocab)}  # inverse table
    wtoi['<eos>'] = 0
    itow[0] = '<eos>'
    wtoi['<sos>'] = 1
    itow[1] = '<sos>'

    dataset_meta = {'vid_to_meta': vid_to_meta,
                    'split_to_ids': split_to_ids,
                    'ix_to_word': itow,
                    'word_to_ix': wtoi}

    out_path = args.out_json_basepath
    out_path += dataset
    out_path += '.json'
    json.dump(dataset_meta, open(out_path, 'w'))
    print("Wrote dataset meta file to {}".format(out_path))


def process_msvd(dict_path):
    """
    Process_x fns will take dataset meta files and always return 2-tuple of (vid_i, caps_i) and splits
    :param dict_path:
    :return:
    """
    gtdict = pickle.load(open(dict_path, 'rb'), encoding='bytes')
    ids = [i.decode() for i in gtdict.keys()]
    # The split is completely arbitrary for now. Should use what Li Yao had.
    split_to_ids = {'train': [ids[i] for i in range(0, 1500)],
                    'val': [ids[i] for i in range(1500, 1600)],
                    'test': [ids[i] for i in range(1600, 1970)]}
    return gtdict, split_to_ids


if __name__ == '__main__':
    dataset_choices = ['msvd']
    opt = argparse.ArgumentParser()
    opt.add_argument('--dataset', help="Dataset to process.", choices=dataset_choices, default='msvd')
    opt.add_argument('--gtdict', help='/path/to/dict_movieID_caption.pkl (MSVD)', required=True)
    opt.add_argument('--threshold', help="Keep words that appear above a certain threshold frequency.", default=1)
    opt.add_argument('--out_json_basepath', help="Basepath to use for written json file.", default="data/dataset_")
    args = opt.parse_args()
    main(args)

#--gtdict "D:\College\Research\2019 Video Captioning Attack Conference Paper\youtube2text_preprocessed_for_arctic_capgen_vid\youtube2text_iccv15\dict_movieID_caption.pkl"