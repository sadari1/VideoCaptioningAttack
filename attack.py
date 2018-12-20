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

#from warpctc_pytorch import CTCLoss
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
#from carlini_attack_utils import interpolate, resizeDifferentiableNormalize

import argparse
import torch

from utils import *
from ImageCaptioner import ImageCaptioner
from yunjey_image_captioning.build_vocab import Vocabulary
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

#needs: the model, alphabet (not sure what this is), target image, weights?
#The oracle is the model


class CarliniAttack:
    #def __init__(self, oracle, alphabet, image_shape, target, file_weights):
    def __init__(self, oracle, image, target):

        #oracle is the ImageCaptioner class, vocab_path is the path to the vocabulary, image is the image to attack, target is the target caption (for now, in future target can also be
        #the image we want to extract target caption from)

        #image = Image.open(image_path).convert('RGBA') WE ARE PASSING IN IMAGE DIRECTLY

        #get image_shape from image itself
        self.learning_rate = 0.005
        # self.learning_rate = 10
        self.num_iterations = 500
        # self.num_iterations = 100
        self.batch_size = bs = 1
        self.phrase_length = len(target)

        print(image.size()[2:4], end='\n')
        #used to be image_shape, but we can just get the image size itself.
        self.o_imW, self.o_imH = image.size()[2:4]
        print(self.o_imH, self.o_imW)
        self.i_imW = self.o_imW
        self.i_imH = self.o_imH

        self.oracle = oracle

        #Do we really need to reset oracle
        #self.weights = file_weights

        # Variable for adversarial noise, which is added to the image to perturb it
        if torch.cuda.is_available():
            self.delta = Variable(torch.rand((1, self.o_imH, self.o_imW)).cuda(), requires_grad=True)
        else:
            self.delta = Variable(torch.rand((1, self.o_imH, self.o_imW)), requires_grad=True)

        # Optimize on delta and use ctc as criterion
        #ctcloss = CTCLoss()
        self.optimizer = optim.Adam([self.delta],
                                    lr=self.learning_rate,
                                    betas=(0.9, 0.999))

        self.loss = self.oracle.forward(image, target)
        #self.ctcloss = ctcloss
        self.target = target
        #self.converter = utils.strLabelConverter(alphabet, attention=False)

    #def _reset_oracle(self):
    #    self.oracle.load_state_dict(self.weights)

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
            pil_im = imager(tensor.cpu().detach())
            return pil_im


    #Execute uses the image path directly. Fix out_dir later, for now it's the same directory (add an argument for output dir for argparse)
    def execute(self, image_path, out_dir):



        #img_path = images[0]
        bs = self.batch_size
        tensorizer = ToTensor()
        #transformer = dataset.resizeNormalize((imgW, imgH))

        #opens the image and makes it color with transparency mask. Explained here:

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
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),
                                                             (0.229, 0.224, 0.225))])

        #image = load_image(image_path, transform)
        image = Image.open(image_path).convert('RGB')
        image = image.resize([224, 224], Image.LANCZOS)

        image = transform(image).unsqueeze(0)

        original_pil = image

        # First just convert to tensor, we need to use a differentiable resize fn later.
       # image = tensorizer(image)
        # image = transformer(image)

        if torch.cuda.is_available():
            image = image.cuda()

        image = Variable(image)
        original = image

        # Get optimizable version of target and length
        length = Variable(torch.IntTensor(bs))
        text = Variable(torch.IntTensor(bs * 5))  # ????????????????

        #t, l = self.converter.encode(self.target)
        #utils.loadData(length, l)
        #utils.loadData(text, t)

        # This controls the mask to create around the border of fonts.
        # 1.0 = mask away white pixels. ~0.7 = mask closer to font . 0.0 = mask away nothing
        whitespace_mask = (original < 0.7).to(dtype=torch.float32)

        dc = 0.75

        #The attack
        for i in range(self.num_iterations):
            # if i % 200 == 0 and i != 0:
            #     sofar = self.delta.cpu().detach().numpy() * 255.
            #     sofar = np.round(sofar).astype(dtype=np.int8)
            #     sofar = sofar / 255.
            #     self.delta = Variable(torch.tensor(sofar).float().cuda(), requires_grad=True)
            #
            #     self.optimizer = optim.Adam([self.delta],
            #                                 lr=self.learning_rate,
            #                                 betas=(0.9, 0.999))

            apply_delta = torch.clamp(self.delta, min=-dc, max=dc)
            apply_delta = apply_delta * whitespace_mask

            pass_in = torch.clamp(apply_delta + original, min=0.0, max=1.0)

            # Now we need to quantize our tensor down to 8 bit precision.
            # If we don't, a lot of adversarial info is lost when we go from float32 to int8 (0-255, for PIL).
            # This makes the optim converge slower but is necessary so info isn't lost during conversions
            # pass_in = pass_in.to(dtype=torch.uint8)
            # pass_in = pass_in.to(dtype=torch.float32)
            # if i % 100 == 0 and i != 0:
            #     self.delta = self._tensor_to_PIL_im(self.delta)
            #     self.delta = tensorizer(self.delta)
            #
            #     if torch.cuda.is_available():
            #         self.delta = self.delta.cuda()
            #
            #     self.delta = Variable(self.delta, requires_grad=True)

            # pass_in = pass_in.view(1, *pass_in.size())

            # Pass to differentiable resize
            # This would work better if the model was trained with such an end to end architecture

            #            pass_in = pass_in.view(1, *pass_in.size())

            pass_in = pass_in.view(*pass_in.size())

            pass_in = interpolate(pass_in,
                                  size=(self.i_imH, self.i_imW),
                                  mode='bilinear', align_corners=True)

            # self._tensor_to_PIL_im(pass_in[0]).show()

            # Instead use our own differentiable version of PIL resizer
            # transformer = resizeDifferentiableNormalize((imgW, imgH))
            # image = transformer(new_input)
            # if torch.cuda.is_available():
            #     image = image.cuda()
            #
            # image = image.view(1, *image.size())
            # image = Variable(image)


            #Assuming the logits come from the encoder
            logits = self.oracle.encoder(pass_in)

            # Model already restored
            preds_size = Variable(torch.IntTensor([logits.size(0)] * bs))

            #cost = self.ctcloss(logits, text, preds_size, length) / bs
            cost = self.oracle.forward(image_tensor=image.to(device), chosen_caption=self.target) / bs

            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()

            # self._reset_oracle()

            if i % 10 == 0:
                logger.debug("iteration: {}, cost: {}".format(i, cost))
            if i % 100 == 0:
                # See how we're doing
                sim_pred = decode_logits(self.oracle, logits)
                logger.debug("Decoding at iteration {}: {}".format(i, sim_pred))
                if sim_pred == self.target:
                    # We're done
                    logger.debug("Early stop.")
                    break

        #is it encoder?
        self.oracle.encoder.eval()

        #original image captions I assume

        #_, original_im_pred = classify_image_pil(self.oracle, original_pil)
        original_im_pred = captionImage(oracle=self.oracle, image_tensor=original_pil)
        logger.debug("Original image caption: {}".format(original_im_pred))

        #apply the mask
        apply_delta = torch.clamp(self.delta, min=-dc, max=dc)
        apply_delta = apply_delta * whitespace_mask

        pass_in = torch.clamp(apply_delta + original, min=0.0, max=1.0)
        pil_attack_float = self._tensor_to_PIL_im(pass_in, mode='F')
        pil_mask = self._tensor_to_PIL_im(apply_delta, mode='RGBA')
        pil_attack_int = self._tensor_to_PIL_im(pass_in, mode='RGBA')

        #attack image captions
        _, attack_pil_classify = captionImage(oracle=self.oracle, image_tensor= pil_attack_int)
        logger.debug("PIL-based image classify: {}".format(attack_pil_classify))

        pass_in = pass_in.view(1, *pass_in.size())
        pass_in = interpolate(pass_in,
                              size=(self.i_imH, self.i_imW),
                              mode='bilinear', align_corners=True)

        new_attack_input = pass_in

        _, attack_ete_classify = captionImage(oracle=self.oracle, image_tensor= new_attack_input)
        logger.debug("Attacked E-t-E classify: {}".format(attack_ete_classify))

        # original_pil.show()
        # pil_attack_int.show()

        run_id = np.random.randint(999999)
        original_path = os.path.join(out_dir, 'original_{}.jpg'.format(run_id))
        delta_path = os.path.join(out_dir, 'delta_{}.jpg'.format(run_id))
        pil_attack_float_path = os.path.join(out_dir, 'attack_{}.tiff'.format(run_id))
        pil_attack_int_path = os.path.join(out_dir, 'attack_{}.jpg'.format(run_id))
        out_ckpt_path = os.path.join(out_dir, 'CTC-CRNN_{}.pt'.format(run_id))

        original_pil.save(original_path)
        pil_mask.save(delta_path)
        pil_attack_float.save(pil_attack_float_path)
        pil_attack_int.save(pil_attack_int_path)

        torch.save(self.oracle.state_dict(), out_ckpt_path)
        logger.debug("Saved to ID {}".format(run_id))

        pickle.dump((original_im_pred, attack_ete_classify), open(os.path.join(out_dir, 'result.pkl'), 'wb'))


'''
def decode_logits(preds):
    _, preds = preds.max(2)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    preds = preds.transpose(1, 0).contiguous().view(-1)

    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    return raw_pred, sim_pred
    '''

def decode_logits(oracle, preds):

    sampled_ids = oracle.decoder.sample(preds)
    sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = oracle.vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    return sentence


def image_pil_to_logits(oracle, pil_im):
    transformer = dataset.resizeNormalize((imgW, imgH))
    image = transformer(pil_im)
    if torch.cuda.is_available():
        image = image.cuda()

    image = image.view(1, *image.size())
    image = Variable(image)

    preds = oracle(image)
    return preds



def captionImage(oracle, image_tensor):


    feature = oracle.encoder(image_tensor)
    sampled_ids = oracle.decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = self.vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    return sentence


def classify_image_pil(oracle, pil_im):
    preds = image_pil_to_logits(oracle, pil_im)

    _, preds = preds.max(2)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    preds = preds.transpose(1, 0).contiguous().view(-1)

    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    return raw_pred, sim_pred


def classify_image_tensor(oracle, tensor):
    # imager = ToPILImage()
    # pil_im = imager(tensor.cpu())

    preds = oracle(tensor)
    _, preds = preds.max(2)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    preds = preds.transpose(1, 0).contiguous().view(-1)

    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    #
    return raw_pred, sim_pred
    #
    # return classify_image_pil(oracle, pil_im)


def _validate(args):
    pass


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)




def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    r"""Down/up samples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`
    Ripped from master branch of PyTorch, which is not stable yet. Just need this fn.
    https://pytorch.org/docs/master/_modules/torch/nn/functional.html#interpolate
    Not ideal but will work for now.
    """

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError('either size or scale_factor should be defined')
        if size is not None and scale_factor is not None:
            raise ValueError('only one of size or scale_factor should be defined')
        if scale_factor is not None and isinstance(scale_factor, tuple)\
                and len(scale_factor) != dim:
            raise ValueError('scale_factor shape must match input shape. '
                             'Input is {}D, scale_factor size is {}'.format(dim, len(scale_factor)))

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        # math.floor might return float in py2.7
        return [int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)]

    if mode in ('nearest', 'area'):
        if align_corners is not None:
            raise ValueError("align_corners option can only be set with the "
                             "interpolating modes: linear | bilinear | trilinear")
    else:
        if align_corners is None:
            warnings.warn("Default upsampling behavior when mode={} is changed "
                          "to align_corners=False since 0.4.0. Please specify "
                          "align_corners=True if the old behavior is desired. "
                          "See the documentation of nn.Upsample for details.".format(mode))
            align_corners = False

    if input.dim() == 3 and mode == 'bilinear':
        raise NotImplementedError("Got 3D input, but bilinear mode needs 4D input")
    elif input.dim() == 4 and mode == 'bilinear':
        return torch._C._nn.upsample_bilinear2d(input, _output_size(2), align_corners)
    elif input.dim() == 5 and mode == 'bilinear':
        raise NotImplementedError("Got 5D input, but bilinear mode needs 4D input")
    elif input.dim() == 5 and mode == 'trilinear':
        return torch._C._nn.upsample_trilinear3d(input, _output_size(3), align_corners)
    else:
        raise NotImplementedError("Input Error: Only 3D, 4D and 5D input Tensors supported"
                                  " (got {}D) for the modes: nearest | linear | bilinear | trilinear"
    " (got {})".format(input.dim(), mode))