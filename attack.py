import utils

import numpy as np
import torch
import torch.optim as optim
import argparse
import logging
import io
import os
import itertools
import pickle

from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

import torch

from utils import *
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig()
logger = logging.getLogger(__name__)
# Change logging level to info if running experiment, debug otherwise
logger.setLevel(logging.DEBUG)

# SEED = 9
nc = 1
imgH = 40
imgW = 180
num_hidden = 256


class CarliniAttack:
    def __init__(self, oracle, image_path, target):
        """
        :param oracle: ImageCaptioner class
        :param image: sample image to get shape
        :param target: target caption (for now,
        in future target can also be the image we want to extract target caption from)
        """

        tensorize = transforms.ToTensor()
        original_pil = Image.open(image_path)
        image_tensor = tensorize(original_pil).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        image_tensor = Variable(image_tensor)
        original = image_tensor


        #0.1 and c = 0.5 work. c=0.54 and 0.07 LR works even better.
        self.learning_rate = 0.07
        # self.learning_rate = 10
        self.num_iterations = 50000
        # self.num_iterations = 100
        self.batch_size = 1
        self.phrase_length = len(target)
        self.oracle = oracle

        # self.weights = file_weights

        # Variable for adversarial noise, which is added to the image to perturb it
        if torch.cuda.is_available():
            self.delta = Variable(torch.zeros(original.shape).cuda(), requires_grad=True)
        else:
            self.delta = Variable(torch.zeros(original.shape), requires_grad=True)

        self.optimizer = optim.Adam([self.delta],
                                    lr=self.learning_rate,
                                    betas=(0.9, 0.999))
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.20)
        self.input_shape = (224, 224)

        self.target = target

    def _reset_oracle(self):
       self.oracle.load_state_dict(self.weights)

    def _tensor_to_PIL_im(self, tensor, mode='RGB'):
        if mode == 'F':
            # The ToPILImage impelementation for 'F' mode is broken
            # https://github.com/pytorch/vision/issues/448
            tensor = tensor.cpu().detach()
            tensor = tensor.mul(255)
            tensor = np.transpose(tensor.numpy(), (1, 2, 0))
            tensor = tensor.squeeze()

            return Image.fromarray(tensor, mode='F')
        else:
            imager = ToPILImage(mode='RGB')
            tensor = tensor.squeeze()
            pil_im = imager(tensor.cpu().detach())
            return pil_im

    def decode_logits(self, image_tensor):
        enc_logits = self.oracle.encoder(image_tensor)
        sampled_ids = self.oracle.decoder.sample(enc_logits)
        sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)

        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = self.oracle.vocab.idx2word[word_id]
            if word == '<start>':
                continue
            if word == '<end>':
                break
            sampled_caption.append(word)
        sentence = ' '.join(sampled_caption)



        return sentence

    # Execute uses the image path directly. Fix out_dir later, for now it's the same directory (add an argument for output dir for argparse)
    def execute(self, image_path):
        #bs = self.batch_size
        # Split up transforms to mimic real attack steps
        tensorize = transforms.ToTensor()
        m_normalize = transforms.Normalize((0.485, 0.456, 0.406),
                                           (0.229, 0.224, 0.225))
        original_i = load_image(image_path, reshape=False)
        original_i.show()
        original_i = original_i.resize([224, 224], Image.LANCZOS)

        image_tensor = tensorize(original_i).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        image_tensor = Variable(image_tensor)
        sim_pred = self.decode_logits(image_tensor)
        print("Original caption: <start> " + sim_pred + " <end>")
        print("Target caption: " + self.target)

        # opens the image and makes it color with transparency mask. Explained here:

        '''
        1 (1-bit pixels, black and white, stored with one pixel per byte)
        L (8-bit pixels, black and white)
        P (8-bit pixels, mapped to any other mode using a colour palette)
        RGB (3x8-bit pixels, true colour)
        RGBA (4x8-bit pixels, true colour with transparency mask)
        CMYK (4x8-bit pixels, colour separation)
        YCbCr (3x8-bit pixels, colour video format)
        I (32-bit signed integer pixels)
        F (32-bit floating point pixels)
        '''

        #original_pil.show()
        original_pil = Image.open(image_path)
        image_tensor = tensorize(original_pil).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        image_tensor = Variable(image_tensor)
        original = image_tensor

        # dc = 0.80
        dc = 1.0
        norm_weight = 0.02

        #c is some constant between 0 and 1

        #c = 0.5
        c=0.54
        # The attack
        for i in range(self.num_iterations):

            apply_delta = torch.clamp(self.delta, min=-dc, max=dc)

            #This below is I + omega)
            pass_in = torch.clamp(apply_delta + original, min=0.0, max=1.0)
            pass_in = torch.nn.functional.interpolate(pass_in, size=self.input_shape, mode='bilinear')
            #pass_in = torch.nn.functional.interpolate(pass_in, self.input_shape, 'trilinear')
            # pass_in = m_normalize(pass_in.squeeze()).unsqueeze(0)
            pass_in = pass_in.view(*pass_in.size())
            pass_in.to(device)

            #This below is loss()
            #loss(I + δ) + L2(I + δ)


            # apply_delta.norm() is L2(I + δ)
            #c * loss(I + omega) + ||omega||2 2
            # ||omega||22 = ||(I+omega) - I||22

            cost = self.oracle.forward(pass_in, self.target)

            y = torch_arctanh(torch.nn.functional.interpolate(original, size=self.input_shape, mode='bilinear'))
            w = torch_arctanh(pass_in) - y
            normterm = (w+y).tanh() - y.tanh()
            cost = c * cost + normterm.norm()


            #minimize w:
            #c * loss (tanh(w + y)) + || tanh(w + y) - tanh(y) || 2 2
            # w = arctanh(I + omega) - y
            # y = arctanh(I)


            #omega is apply_delta, original is I

            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()

            logger.debug("iteration: {}, cost: {}".format(i, cost))
            adv_sample = torch.clamp(apply_delta + original, min=0.0, max=1.0)
            sim_pred = self.decode_logits(pass_in)

            if sim_pred == self.target:
                # We're done
                logger.debug("Decoding at iteration {}: <start> {} <end>".format(i, sim_pred))
                logger.debug("Early stop. Cost: {}".format(cost))
                self._tensor_to_PIL_im(adv_sample).show()
                break


            if i % 10 == 0:
                # See how we're doing
                logger.debug("Decoding at iteration {}: <start> {} <end>".format(i, sim_pred))

                if sim_pred == self.target:
                    # We're done
                    logger.debug("Early stop.")
                    self._tensor_to_PIL_im(adv_sample).show()
                    break

            if i % 100 == 0:
                self._tensor_to_PIL_im(adv_sample).show()


        self.oracle.encoder.eval()
        self.oracle.decoder.eval()

        print(image_path)
        adv_image = self._tensor_to_PIL_im(adv_sample)
        imgpath = image_path.split('/')
        advpath = '' + imgpath[0]
        for i in range(1, len(imgpath)-1):
            advpath += '/%s' % imgpath[i]

        print(advpath)
        filename = imgpath[len(imgpath)-1].split('.')
        advpath += '/%s_adversarial.%s' % (filename[0], filename[1])
        adv_image.save(advpath)
        print(advpath)



# def captionImage(oracle, image_tensor):
#
#
#     feature = oracle.encoder(image_tensor)
#     sampled_ids = oracle.decoder.sample(feature)
#     sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)
#
#     # Convert word_ids to words
#     sampled_caption = []
#     for word_id in sampled_ids:
#         word = self.vocab.idx2word[word_id]
#         sampled_caption.append(word)
#         if word == '<end>':
#             break
#     sentence = ' '.join(sampled_caption)
#
#     return sentence


# def classify_image_pil(oracle, pil_im):
#     preds = image_pil_to_logits(oracle, pil_im)
#
#     _, preds = preds.max(2)
#
#     preds_size = Variable(torch.IntTensor([preds.size(0)]))
#     preds = preds.transpose(1, 0).contiguous().view(-1)
#
#     raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
#     sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
#
#     return raw_pred, sim_pred

#
# def classify_image_tensor(oracle, tensor):
#     # imager = ToPILImage()
#     # pil_im = imager(tensor.cpu())
#
#     preds = oracle(tensor)
#     _, preds = preds.max(2)
#
#     preds_size = Variable(torch.IntTensor([preds.size(0)]))
#     preds = preds.transpose(1, 0).contiguous().view(-1)
#
#     raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
#     sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
#     #
#     return raw_pred, sim_pred
#     #
#     # return classify_image_pil(oracle, pil_im)


def _validate(args):
    pass

def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#
# def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
#     r"""Down/up samples the input to either the given :attr:`size` or the given
#     :attr:`scale_factor`
#     Ripped from master branch of PyTorch, which is not stable yet. Just need this fn.
#     https://pytorch.org/docs/master/_modules/torch/nn/functional.html#interpolate
#     Not ideal but will work for now.
#     """
#
#     def _check_size_scale_factor(dim):
#         if size is None and scale_factor is None:
#             raise ValueError('either size or scale_factor should be defined')
#         if size is not None and scale_factor is not None:
#             raise ValueError('only one of size or scale_factor should be defined')
#         if scale_factor is not None and isinstance(scale_factor, tuple)\
#                 and len(scale_factor) != dim:
#             raise ValueError('scale_factor shape must match input shape. '
#                              'Input is {}D, scale_factor size is {}'.format(dim, len(scale_factor)))
#
#     def _output_size(dim):
#         _check_size_scale_factor(dim)
#         if size is not None:
#             return size
#         scale_factors = _ntuple(dim)(scale_factor)
#         # math.floor might return float in py2.7
#         return [int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)]
#
#     if mode in ('nearest', 'area'):
#         if align_corners is not None:
#             raise ValueError("align_corners option can only be set with the "
#                              "interpolating modes: linear | bilinear | trilinear")
#     else:
#         if align_corners is None:
#             warnings.warn("Default upsampling behavior when mode={} is changed "
#                           "to align_corners=False since 0.4.0. Please specify "
#                           "align_corners=True if the old behavior is desired. "
#                           "See the documentation of nn.Upsample for details.".format(mode))
#             align_corners = False
#
#     if input.dim() == 3 and mode == 'bilinear':
#         raise NotImplementedError("Got 3D input, but bilinear mode needs 4D input")
#     elif input.dim() == 4 and mode == 'bilinear':
#         return torch._C._nn.upsample_bilinear2d(input, _output_size(2), align_corners)
#     elif input.dim() == 5 and mode == 'bilinear':
#         raise NotImplementedError("Got 5D input, but bilinear mode needs 4D input")
#     elif input.dim() == 5 and mode == 'trilinear':
#         return torch._C._nn.upsample_trilinear3d(input, _output_size(3), align_corners)
#     else:
#         raise NotImplementedError("Input Error: Only 3D, 4D and 5D input Tensors supported"
#                                   " (got {}D) for the modes: nearest | linear | bilinear | trilinear"
#     " (got {})".format(input.dim(), mode))