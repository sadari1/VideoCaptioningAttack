import math
import torch
import torch.optim as optim
import logging
import skvideo.io
from torch.autograd import Variable
from torchvision import transforms
from global_constants import *
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np

logging.basicConfig()
logger = logging.getLogger(__name__)
# Change logging level to info if running experiment, debug otherwise
logger.setLevel(logging.DEBUG)

#save as numpy file and try

class S2VT_Attack:
    def __init__(self, model, video_path, target, dataset, config, optimizer, crit, seq_decoder, window=None ):
        # self.seed = config.seed

        self.video_path = video_path
        self.config = config
        self.batch_size = config["batch_size"]
        if window:
            self.frames = skvideo.io.vread(video_path,num_frames=config["num_frames"])[window[0]:window[-1] + 1]
        else:
            self.frames = skvideo.io.vread(video_path,num_frames=config["num_frames"])[0:self.batch_size]

        # self.loss = loss

        self.seq_decoder = seq_decoder
        # dc = 0.80
        self.c = config["c"]
        self.dc = 255

        self.num_iterations = config["num_iterations"]
        self.phrase_length = len(target)
        self.model = model
        self.dataset = dataset
        self.vocab = dataset.get_vocab()

        if torch.cuda.is_available():
            self.delta = Variable(torch.zeros(self.frames.shape).cuda(), requires_grad=True)
        else:
            self.delta = Variable(torch.zeros(self.frames.shape), requires_grad=True)

        #optimizer is a list. Optimizer[0] is the function name, optimizer[1] is the parameters.
        if optimizer[0] == 'Adam':
            self.optimizer = optim.Adam([self.delta], lr=config["learning_rate"], betas=optimizer[1])

        self.crit = crit

        self.input_shape = config["input_shape"]
        self.target = target
        self.real_target = ' '.join(target.split(' ')[1:-1])

        self.tlabel, self.tmask = self.produce_t_mask()

    def loss(self, seq_prob):
        loss = self.crit(seq_prob, self.tlabel[:, 1:].cuda(), self.tmask[:, 1:].cuda())
        return loss

    def produce_t_mask(self):
        mask = torch.zeros(self.dataset.max_len)
        captions = [self.target.split(' ')]
        gts = torch.zeros(len(captions), self.dataset.max_len).long()
        for i, cap in enumerate(captions):
            if len(cap) > self.dataset.max_len:
                cap = cap[:self.dataset.max_len]
                cap[-1] = '<eos>'
            for j, w in enumerate(cap):
                gts[i, j] = self.dataset.word_to_ix[w]

        label = gts[0]
        non_zero = (label == 0).nonzero()
        mask[:int(non_zero[0]) + 1] = 1

        return label.unsqueeze(0), mask.unsqueeze(0)


    def original(self):

        print("Frame length: {}".format(len(self.frames)))

        original = torch.tensor(self.frames)
        original = (original.float()).cuda()

        with torch.no_grad():
            # bp ---
            batch = self.create_batches(frames_to_do=original)
            seq_prob, seq_preds = self.model(batch.unsqueeze(0), mode='inference')
            sents = self.seq_decoder(self.vocab, seq_preds)

            for sent in sents:
                print('Original caption: ' + sent)

        print('Target caption: ' + self.target)
        return original

    def costs_decoded(self, original):
        apply_delta = torch.clamp(self.delta * 255., min=-self.dc, max=self.dc)

        # This too
        # The perturbation is applied to the original and resized through interpolation
        pass_in = torch.clamp(apply_delta + original, min=0, max=255)

        # Put this in the function as well to decode at each iteration
        batch = self.create_batches(pass_in)
        feats = self.model.conv_forward(batch.unsqueeze(0))
        seq_prob, seq_preds = self.model.encoder_decoder_forward(feats, mode='inference')
        cost = self.loss(seq_prob=seq_prob)
        sents = self.seq_decoder(self.vocab, seq_preds)

        return cost, sents[0], apply_delta, pass_in

    def save_to(self, adv_path, apply_delta, pass_in):
        def save_tensor_to_video(batched_t, fpath):
            in_frames = batched_t.detach().cpu().numpy()
            skvideo.io.vwrite(fpath, in_frames)

        def plt_collate_batch(batched_t):
            n_col = 2
            n_rows = np.max([2, int(len(batched_t) / n_col)])
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_col, sharey=True, sharex=True)

            k = 0
            for i in range(n_rows):
                for j in range(n_col):
                    if (k >= len(batched_t)):
                        showing = torch.from_numpy(np.ones(batched_t[0].shape))
                    else:
                        showing = batched_t[k]

                    if batched_t.shape[-1] == 3:
                        showing = showing.detach().cpu().numpy()
                    else:
                        showing = showing.permute(1, 2, 0).detach().cpu().numpy()

                    axes[i, j].imshow(showing)
                    k += 1

            plt.tight_layout()
            plt.show()


        plt_collate_batch(pass_in / 255.)
        logger.info("Saving adversarial video to:\t{}".format(adv_path))
        save_tensor_to_video(pass_in, adv_path)
        save_tensor_to_video(apply_delta, 'perturbation_' + adv_path)


    def plt_collate_batch(self, batched_t):
        n_col = 2
        n_rows = np.max([2, int(len(batched_t) / n_col)])
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_col, sharey=True, sharex=True)

        k = 0
        for i in range(n_rows):
            for j in range(n_col):
                if (k >= len(batched_t)):
                    showing = torch.from_numpy(np.ones(batched_t[0].shape))
                else:
                    showing = batched_t[k]

                if batched_t.shape[-1] == 3:
                    showing = showing.detach().cpu().numpy()
                else:
                    showing = showing.permute(1, 2, 0).detach().cpu().numpy()

                axes[i, j].imshow(showing)
                k += 1

        plt.tight_layout()
        plt.show()


    def create_batches(self, frames_to_do):
        batch_size = self.batch_size
        class ToSpaceBGR(object):

            def __init__(self, is_bgr):
                self.is_bgr = is_bgr

            def __call__(self, tensor):
                if self.is_bgr:
                    new_tensor = tensor.clone()
                    new_tensor[0] = tensor[2]
                    new_tensor[2] = tensor[0]
                    tensor = new_tensor
                return tensor

        class ToRange255(object):

            def __init__(self, is_255):
                self.is_255 = is_255

            def __call__(self, tensor):
                if self.is_255:
                    tensor.mul_(255)
                return tensor

        n = frames_to_do.shape[0]
        h, w = frames_to_do.shape[1:3]
        scale = 0.875
        dim = self.config["dimensions"]
        input_size = [3, dim, dim]
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        input_range = [0, 1.]
        input_space = 'RGB'
        expand_size = int(math.floor(max(input_size) / scale))
        if w < h:
            ow = expand_size
            oh = int(expand_size * h / w)
        else:
            oh = expand_size
            ow = int(expand_size * w / h)

        tfs = []
        tfs.append(ToSpaceBGR(input_space == 'BGR'))
        tfs.append(ToRange255(max(input_range) == 255))
        tfs.append(transforms.Normalize(mean=mean, std=std))
        tf = transforms.Compose(tfs)

        a = int((0.5 * oh) - (0.5 * float(input_size[1])))
        b = a + input_size[1]
        c = int((0.5 * ow) - (0.5 * float(input_size[2])))
        d = c + input_size[2]

        if n < batch_size:
            logger.warning("Sample size less than batch size: Cutting batch size.")
            batch_size = n

        logger.info("Generating {} batches...".format(n // batch_size))

        for idx in range(0, n, batch_size):
            frames_idx = list(range(idx, min(idx + batch_size, n)))

            # <batch, h, w, ch> <0,255>
            batch_tensor = frames_to_do[frames_idx]

            pass_in = batch_tensor.permute(0, 3, 1, 2) / 255.
            inp = torch.nn.functional.interpolate(pass_in,
                                                  size=(oh, ow),
                                                  mode='bilinear', align_corners=True)
            # Center cropping
            cropped_frames = inp[:, :, a:b, c:d]
            # cropped_image = cropped_image.contiguous()
            for i in range(len(cropped_frames)):
                cropped_frames[i] = tf(cropped_frames[i])
        return cropped_frames
