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
    def __init__(self, oracle, image, target):
        """
        :param oracle: ImageCaptioner class
        :param image: sample image to get shape
        :param target: target caption (for now,
        in future target can also be the image we want to extract target caption from)
        """

        self.learning_rate = 0.2
        # self.learning_rate = 10
        self.num_iterations = 50000
        # self.num_iterations = 100
        self.batch_size = 1
        self.phrase_length = len(target)
        self.oracle = oracle

        # self.weights = file_weights

        # Variable for adversarial noise, which is added to the image to perturb it
        if torch.cuda.is_available():
            self.delta = Variable(torch.zeros(image.shape).cuda(), requires_grad=True)
        else:
            self.delta = Variable(torch.zeros(image.shape), requires_grad=True)

        self.optimizer = optim.Adam([self.delta],
                                    lr=self.learning_rate,
                                    betas=(0.9, 0.999))
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.20)


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
    def execute(self, image_path, out_dir):
        bs = self.batch_size

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

        # Split up transforms to mimic real attack steps
        tensorize = transforms.ToTensor()
        m_normalize = transforms.Normalize((0.485, 0.456, 0.406),
                                           (0.229, 0.224, 0.225))

        original_pil = load_image(image_path, None)
        # original_pil.show()

        image_tensor = tensorize(original_pil).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        image_tensor = Variable(image_tensor)
        original = image_tensor

        # self.oracle.encoder.eval()

        sim_pred = self.decode_logits(original)
        print("Original caption: <start> " + sim_pred + " <end>")
        print("Target caption: " + self.target)
        self._tensor_to_PIL_im(original).show()

        # This controls the mask to create around the border of fonts.
        # 1.0 = mask away white pixels. ~0.7 = mask closer to font . 0.0 = mask away nothing
        # whitespace_mask = (original < 0.7).to(dtype=torch.float32)

        # dc = 0.80
        dc = 1.0
        norm_weight = 0.02

        #c is some constant between 0 and 1

        c = 0.5
        # The attack
        for i in range(self.num_iterations):

            apply_delta = torch.clamp(self.delta, min=-dc, max=dc)
            # apply_delta = apply_delta * whitespace_mask

            #This below is I + omega)
            pass_in = torch.clamp(apply_delta + original, min=0.0, max=1.0)
            #pass_in = m_normalize(pass_in.squeeze()).unsqueeze(0)  # messy, but start of oracle preproc

            pass_in = pass_in.view(*pass_in.size())
            #pass_in = torch.clamp(pass_in, min=0.0, max=1.0)
            pass_in.to(device)

            #This below is loss()
            #cost = self.oracle.forward(pass_in, self.target)
            #cost = cost / bs
            #loss(I + δ) + L2(I + δ)


            # apply_delta.norm() is L2(I + δ)
            #cost = ((1 - norm_weight) * cost) + (norm_weight * apply_delta.norm())

            #cost = cost + apply_delta.norm()

            #cost = c * cost + apply_delta.norm()

            #c * loss(I + omega) + ||omega||2 2
            # ||omega||22 = ||(I+omega) - I||22

            #c * cost + apply_delta.norm())

            #n = val.detach().clone().requires_grad_(True)


            #y = np.arctanh(original.detach().cpu().clone().requires_grad_(True).numpy())
            #y = torch.arctanh(original)
            #y = np.arctanh(original)


            #y = 0.5 * torch.log((1 + original) / (1 - original))
            y = torch_arctanh(original)
            w = torch_arctanh(pass_in) - y

            #0.5 * torch.log((1 + y) / (1 - y))
            #w = arctanh(pass_in) - y
            #w = np.arctanh(pass_in) - y
            #w  = 0.5 * torch.log((1 + pass_in) / (1 - pass_in)) - y
            #w = np.arctanh(pass_in.detach().cpu().clone().requires_grad_(True).numpy()) - y

            #print(w, y)
            #passing in w + y, cancels out the -y in w, so it's just w without the - y
            cost = self.oracle.forward((w+y).tanh(), self.target)


            #print(cost)
            #normterm = torch.norm((w+y).tanh() - (y).tanh())
            #normterm = torch.norm((w+y).tanh() - y.tanh(), 2)

            #print((w+y).tanh())
            #print('-----\n\n', y.tanh(), (w+y).tanh() - y.tanh())
            #a = (w+y).tanh() - y.tanh()
            #normterm = (a.pow(2).sqrt()).sum()

            #a = (w+y).tanh() - y.tanh()
            #normterm = a.min() ** 2
            #normterm = (a.pow(2).sqrt()).sum()

            normterm = (w+y).tanh() - y.tanh()
            #print(normterm, normterm.sum())
            cost = c * cost + normterm.norm()


            #minimize w:
            #c * loss (tanh(w + y)) + || tanh(w + y) - tanh(y) || 2 2
            # w = arctanh(I + omega) - y
            # y = arctanh(I)


            #omega is apply_delta, original is I

            # artanh(y) = 0.5*(torch.log(1+y)/(1-y))


            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()
            # self.scheduler.step()

            # self._reset_oracle()

            #if i % 10 == 0:
            logger.debug("iteration: {}, cost: {}".format(i, cost))
            adv_sample = torch.clamp(apply_delta + original, min=0.0, max=1.0)
            sim_pred = self.decode_logits(adv_sample)

            if sim_pred == self.target:
                # We're done
                logger.debug("Decoding at iteration {}: <start> {} <end>".format(i, sim_pred))
                logger.debug("Early stop. Cost: {}".format(cost))
                self._tensor_to_PIL_im(adv_sample).show()

                break


            if i % 100 == 0:
                # See how we're doing
                #adv_sample = torch.clamp(apply_delta + original, min=0.0, max=1.0)
                #sim_pred = self.decode_logits(adv_sample)
                logger.debug("Decoding at iteration {}: <start> {} <end>".format(i, sim_pred))
                #print('DEBUG: \n\n', original, ' \n ', y, '\n\n', pass_in, '\n\n', w)
                #print(torch.isnan(w), '\n\n', torch.isnan(y))

                #if i % 300 == 0:

                if sim_pred == self.target:
                    # We're done
                    logger.debug("Early stop.")
                    self._tensor_to_PIL_im(adv_sample).show()

                    break

        self.oracle.encoder.eval()
        self.oracle.decoder.eval()

        # original image captions I assume
        # TODO: Make usable eventually
        # #_, original_im_pred = classify_image_pil(self.oracle, original_pil)
        # original_im_pred = captionImage(oracle=self.oracle, image_tensor=original_pil)
        # logger.debug("Original image caption: {}".format(original_im_pred))
        #
        # #apply the mask
        # apply_delta = torch.clamp(self.delta, min=-dc, max=dc)
        # # apply_delta = apply_delta * whitespace_mask
        #
        # pass_in = torch.clamp(apply_delta + original, min=0.0, max=1.0)
        # pil_attack_float = self._tensor_to_PIL_im(pass_in, mode='F')
        # pil_mask = self._tensor_to_PIL_im(apply_delta, mode='RGBA')
        # pil_attack_int = self._tensor_to_PIL_im(pass_in, mode='RGBA')
        #
        # #attack image captions
        # _, attack_pil_classify = captionImage(oracle=self.oracle, image_tensor= pil_attack_int)
        # logger.debug("PIL-based image classify: {}".format(attack_pil_classify))
        #
        # pass_in = pass_in.view(1, *pass_in.size())
        # pass_in = interpolate(pass_in,
        #                       size=(self.i_imH, self.i_imW),
        #                       mode='bilinear', align_corners=True)
        #
        # new_attack_input = pass_in
        #
        # _, attack_ete_classify = captionImage(oracle=self.oracle, image_tensor= new_attack_input)
        # logger.debug("Attacked E-t-E classify: {}".format(attack_ete_classify))
        #
        # # original_pil.show()
        # # pil_attack_int.show()
        #
        # run_id = np.random.randint(999999)
        # original_path = os.path.join(out_dir, 'original_{}.jpg'.format(run_id))
        # delta_path = os.path.join(out_dir, 'delta_{}.jpg'.format(run_id))
        # pil_attack_float_path = os.path.join(out_dir, 'attack_{}.tiff'.format(run_id))
        # pil_attack_int_path = os.path.join(out_dir, 'attack_{}.jpg'.format(run_id))
        # out_ckpt_path = os.path.join(out_dir, 'CTC-CRNN_{}.pt'.format(run_id))
        #
        # original_pil.save(original_path)
        # pil_mask.save(delta_path)
        # pil_attack_float.save(pil_attack_float_path)
        # pil_attack_int.save(pil_attack_int_path)
        #
        # torch.save(self.oracle.state_dict(), out_ckpt_path)
        # logger.debug("Saved to ID {}".format(run_id))
        #
        # pickle.dump((original_im_pred, attack_ete_classify), open(os.path.join(out_dir, 'result.pkl'), 'wb'))



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


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image

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