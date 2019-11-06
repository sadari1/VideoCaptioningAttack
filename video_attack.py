import os
import math
import numpy as np
import torch
import torch.optim as optim
import logging
import skvideo.io
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import PIL
from pretrainedmodels import utils as ptm_utils
from video_caption_pytorch.models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from video_caption_pytorch.dataloader import VideoDataset
from pretrainedmodels import utils as ptm_utils
from video_caption_pytorch.models.ConvS2VT import ConvS2VT
from video_caption_pytorch.misc import utils as utils
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from torchvision import transforms
from utils import *
from global_constants import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig()
logger = logging.getLogger(__name__)
# Change logging level to info if running experiment, debug otherwise
logger.setLevel(logging.DEBUG)

BATCH_SIZE = 3
c = 100


class CarliniAttack:
    def __init__(self, oracle, video_path, target, dataset, att_window = None, window=None, seed=SEED):
        """
        :param oracle: ImageCaptioner class
        :param image: sample image to get shape
        :param target: target caption (for now,
        in future target can also be the image we want to extract target caption from)
        """
        self.seed = seed

        if att_window:
            frames = []
            for f in att_window:
                frames.append(skvideo.io.vread(video_path)[f])

        if window:
            frames = skvideo.io.vread(video_path)[window[0]:window[-1] + 1]
        else:
            frames = skvideo.io.vread(video_path)[0:BATCH_SIZE]
            # frames = skvideo.io.vread(video_path)[3:6]

        # 0.001 -> smaller perturbations
        self.learning_rate = 0.005
        # self.learning_rate = 10
        self.num_iterations = 1000
        # self.num_iterations = 100
        self.batch_size = 1
        self.phrase_length = len(target)
        self.oracle = oracle
        self.dataset = dataset
        self.vocab = dataset.get_vocab()
        # Variable for adversarial noise, which is added to the image to perturb it
        # Starts as an empty mask so noise will be added onto it
        if torch.cuda.is_available():
            self.delta = Variable(torch.zeros(frames.shape).cuda(), requires_grad=True)
        else:
            self.delta = Variable(torch.zeros(frames.shape), requires_grad=True)

        self.optimizer = optim.Adam([self.delta],
                                    lr=self.learning_rate,
                                    betas=(0.9, 0.999))

        self.crit = utils.LanguageModelCriterion()

        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.20)
        self.input_shape = (299, 299)
        self.target = target
        self.real_target = ' '.join(target.split(' ')[1:-1])
        # self.real_target = target
        self.tlabel, self.tmask = self.produce_t_mask()

        del(frames)
        torch.cuda.empty_cache()

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

    def loss(self, seq_prob):
        loss = self.crit(seq_prob, self.tlabel[:, 1:].cuda(), self.tmask[:, 1:].cuda())
        return loss

    # Execute uses the image path directly. Fix out_dir later, for now it's the same directory (add an argument for output dir for argparse)
    def execute(self, video_path, att_window = None, window=None, functional=False, stats=False):

        print(video_path)

        if att_window:
            frames = []
            for f in att_window:
                frames.append(skvideo.io.vread(video_path)[f])
            print("Attention frame length: {}".format(len(frames)))

        if window:
            frames = skvideo.io.vread(video_path)[window[0]:window[-1] + 1]
            print("Frame length: {}".format(len(frames)))
        else:
            frames = skvideo.io.vread(video_path)[0:BATCH_SIZE]
            # frames = skvideo.io.vread(video_path)[BATCH_SIZE:BATCH_SIZE+3]
        # plt.imshow(frames[0])
        # plt.show()
        original = torch.tensor(frames)
        original = (original.float()).cuda()

        with torch.no_grad():
            # bp ---
            batch = create_batches(original)
            seq_prob, seq_preds = self.oracle(batch.unsqueeze(0), mode='inference')
            sents = utils.decode_sequence(self.vocab, seq_preds)

            for sent in sents:
                print('Original caption: ' + sent)

        print('Target caption: ' + self.target)

        # all the frames are stored in batches. Batches[0] should contain the first 32 frames.
        torch.cuda.empty_cache()
        # dc = 0.80
        dc = 255


        # The attack
        for i in range(self.num_iterations):

            apply_delta = torch.clamp(self.delta * 255., min=-dc, max=dc)

            # The perturbation is applied to the original and resized through interpolation
            pass_in = torch.clamp(apply_delta + original, min=0, max =255)

            batch = create_batches(pass_in)
            feats = self.oracle.conv_forward(batch.unsqueeze(0))
            seq_prob, seq_preds = self.oracle.encoder_decoder_forward(feats, mode='inference')

            cost = self.loss(seq_prob)
            sents = utils.decode_sequence(self.vocab, seq_preds)
            logger.info("Decoding at iteration {}: {} ".format(i, sents[0]))

            if sents[0] == self.real_target or i == (self.num_iterations - 1):

                # We're done
                logger.debug("Decoding at iteration {}:\t{} ".format(i, sents[0]))
                logger.debug("Early stop. Cost: {}".format(cost))

                base_toks = video_path.split('/')
                base_dir_toks = base_toks[:-1]
                base_filename = base_toks[-1]
                base_name = ''.join(base_filename.split('.')[:-1])
                adv_path = os.path.join('/'.join(base_dir_toks), base_name + '_adversarial.avi')

                # plt_tensor(pass_in/255.)
                if not functional:
                    plt_collate_batch(pass_in / 255.)
                    logger.info("Saving adversarial video to:\t{}".format(adv_path))
                    save_tensor_to_video(pass_in, adv_path)
                    save_tensor_to_video(apply_delta, 'perturbation_' + adv_path)

                if stats:
                    return {'pass_in': pass_in.detach().cpu().numpy(),
                            'iterations': i,
                            'delta': self.delta.detach().cpu().numpy()}
                else:
                    return pass_in

            # Every 10 iterations it outputs the caption.
            if i % 1000 == 0:
                # See how we're doing
                logger.info("Decoding at iteration {}: {} ".format(i, sents[0]))
                if not functional:
                    plt_collate_batch(pass_in / 255.)

            # if i % 20 == 0:
            #     plt_tensor(pass_in/255.)

            # print("Norm:\t{}\t\tCost:\t{}".format(apply_delta.norm(), cost.data))

            # w and y make calculations more efficient and are used to calculate the l2 norm
            y = torch_arctanh(original / 255.).cuda()
            w = torch_arctanh(pass_in / 255.) - y
            normterm = ((w+y).tanh() - y.tanh())
            normterm = normterm.mean(0).norm()
            # normterm = self.delta / 255.
            # normterm = normterm.mean(0).norm()
            # cost = (c * cost.tanh() + 1) + ((1 - c) * normterm.mean(0).norm().tanh() + 1)
            print("Cost:\t{}\t+\tNormterm:\t{}".format(cost, normterm))
            cost = cost + (c * normterm)

            # calculate gradients
            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()

            # Iteration and cost displayed at every step. We apply the perturbation to the original image again to find the adversarial caption.
            logger.debug("\nIteration: {}, cost: {}".format(i, cost))
            # torch.cuda.empty_cache()

            # Every iteration it checks for whether or not the target caption equals the original


def PIL_to_image(image_path):
    tensorize = transforms.ToTensor()
    original_pil = Image.open(image_path)
    image_tensor = tensorize(original_pil).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    image_tensor = Variable(image_tensor)
    return image_tensor


def save_tensor_to_video(batched_t, fpath):
    in_frames = batched_t.detach().cpu().numpy()
    skvideo.io.vwrite(fpath, in_frames)


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


def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5


def plt_tensor(batched_t):
    showing = batched_t[0]
    if batched_t.shape[-1] == 3:
        showing = showing.detach().cpu().numpy()
    else:
        showing = showing.permute(1, 2, 0).detach().cpu().numpy()

    plt.imshow(showing)
    plt.show()


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


def create_batches(frames_to_do, batch_size=BATCH_SIZE):
    n = frames_to_do.shape[0]
    h, w = frames_to_do.shape[1:3]
    scale = 0.875
    input_size = [3, 331, 331]
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    input_range = [0,1.]
    input_space='RGB'
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

    for idx in range (0, n, batch_size):
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


def nondiff_create_batches(frames_to_do, batch_size=BATCH_SIZE):
    n = frames_to_do.shape[0]
    h, w = frames_to_do.shape[1:3]
    scale = 0.875
    input_size = [3, 331, 331]
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    input_range = [0,1.]
    input_space='RGB'
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

    batches = []

    for idx in range (0, n, batch_size):
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

        batches.append(cropped_frames)

    return np.concatenate(batches, axis=0)


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()