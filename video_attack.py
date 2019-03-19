import os
import math
import numpy as np
import torch
import torch.optim as optim
import logging
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from torchvision import transforms
from utils import *
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig()
logger = logging.getLogger(__name__)
# Change logging level to info if running experiment, debug otherwise
logger.setLevel(logging.DEBUG)

BATCH_SIZE = 4


class CarliniAttack:
    def __init__(self, oracle, video_path, target, dataset):
        """
        :param oracle: ImageCaptioner class
        :param image: sample image to get shape
        :param target: target caption (for now,
        in future target can also be the image we want to extract target caption from)
        """
        self.seed = 9
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        frames = skvideo.io.vread(video_path)[0:BATCH_SIZE]
        #0.1 and c = 0.5 work. c=0.54 and 0.07 LR works even better.
        self.learning_rate = 0.001
        # self.learning_rate = 10
        self.num_iterations = 50000
        # self.num_iterations = 100
        self.batch_size = 1
        self.phrase_length = len(target)
        self.oracle = oracle
        self.dataset = dataset
        self.vocab = dataset.get_vocab()
        # Variable for adversarial noise, which is added to the image to perturb it
        # Starts as an empty mask so noise will be added onto it
        if torch.cuda.is_available():
            #TODO
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
        loss = self.crit(seq_prob.unsqueeze(0), self.tlabel[:, 1:].cuda(), self.tmask[:, 1:].cuda())
        return loss

    # Execute uses the image path directly. Fix out_dir later, for now it's the same directory (add an argument for output dir for argparse)
    def execute(self, video_path):

        print(video_path)
        frames = skvideo.io.vread(video_path)[:BATCH_SIZE]
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

        c = 0.99
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

            # print("Norm:\t{}\t\tCost:\t{}".format(apply_delta.norm(), cost.data))

            # w and y make calculations more efficient and are used to calculate the l2 norm
            y = torch_arctanh(original / 255.).cuda()
            w = torch_arctanh(pass_in / 255.) - y
            normterm = ((w+y).tanh() - y.tanh())
            normterm = torch.abs(torch.sigmoid(normterm.mean(0).norm()) - 0.5)
            # cost = (c * cost.tanh() + 1) + ((1 - c) * normterm.mean(0).norm().tanh() + 1)
            print("Cost:\t{}\t+\tNormterm:\t{}".format(cost, normterm))
            cost = (c * cost) + ((1 - c) * normterm)

            # cost = cost * 255.s

            # calculate gradients
            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()

            # Iteration and cost displayed at every step. We apply the perturbation to the original image again to find the adversarial caption.
            logger.debug("\nIteration: {}, cost: {}".format(i, cost))
            # torch.cuda.empty_cache()

            # Every iteration it checks for whether or not the target caption equals the original
            if sents[0] == self.real_target:
                # We're done
                logger.debug("Decoding at iteration {}:\t{} ".format(i, sents[0]))
                logger.debug("Early stop. Cost: {}".format(cost))
                plt_collate_batch(pass_in / 255.)

                base_toks = video_path.split('/')
                base_dir_toks = base_toks[:-1]
                base_filename = base_toks[-1]
                base_name = ''.join(base_filename.split('.')[:-1])
                adv_path = os.path.join('/'.join(base_dir_toks), base_name + '_adversarial.avi')

                logger.info("Saving adversarial video to:\t{}".format(adv_path))

                save_tensor_to_video(pass_in, adv_path)
                break

            # Every 10 iterations it outputs the caption.
            if i % 1000 == 0:
                # See how we're doing
                logger.info("Decoding at iteration {}: {} ".format(i, sents[0]))
                plt_collate_batch(pass_in / 255.)


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

            ## Compare this code with that in line 81 of functional.py in torchvision/transforms.
            #Runs line 70, goes to 78 and runs until 83.

            # cropped_image[i] = cropped_image[i].reshape(331, 331, 3).transpose(0, 1).transpose(0, 2).contiguous()
            cropped_frames[i] = tf(cropped_frames[i])

            # img = img.transpose(0, 1).transpose(0, 2).contiguous()
            # img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            # img = img.view(pic.size[1], pic.size[0], nchannel)
            # # put it from HWC to CHW format
            # # yikes, this transpose takes 80% of the loading time/CPU
            # img = img.transpose(0, 1).transpose(0, 2).contiguous()
            # img.float().div(255)

            # inp = inp.squeeze(0)
            # inp = load_image_fn(frame_.detach().cpu().numpy().astype(np.uint8))
            # input_tensor = tf_img_fn(inp)  # 3x400x225 -> 3x299x299 size may differ
            # input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
            # batch_tensor[i] = inp
        # batches.append(frames_to_do[frames_idx])

        # batch_ag = torch.autograd.Variable(batch_tensor, requires_grad=False)

    return cropped_frames


# def o_create_batches(frames_to_do, load_img_fn, tf_img_fn, batch_size=BATCH_SIZE):
#     n = len(frames_to_do)
#     if n < batch_size:
#         logger.warning("Sample size less than batch size: Cutting batch size.")
#         batch_size = n
#
#     logger.info("Generating {} batches...".format(n // batch_size))
#     batches = []
#     frames_to_do = np.array(frames_to_do)
#
#     for idx in range(0, n, batch_size):
#         frames_idx = list(range(idx, min(idx+batch_size, n)))
#         batch_frames = frames_to_do[frames_idx]
#
#         batch_tensor = torch.zeros((len(batch_frames),) + tuple(tf_img_fn.input_size))
#         for i, frame_ in enumerate(batch_frames):
#             input_img = load_img_fn(frame_)
#             input_tensor = tf_img_fn(input_img)  # 3x400x225 -> 3x299x299 size may differ
#             # input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
#             batch_tensor[i] = input_tensor
#
#         batch_ag = torch.autograd.Variable(batch_tensor, requires_grad=False)
#         batches.append(batch_ag)
#
#     return batches


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()