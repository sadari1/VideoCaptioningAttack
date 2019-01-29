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

BATCH_SIZE = 16


class CarliniAttack:
    def __init__(self, oracle, video_path, target, dataset):
        """
        :param oracle: ImageCaptioner class
        :param image: sample image to get shape
        :param target: target caption (for now,
        in future target can also be the image we want to extract target caption from)
        """

        frames = skvideo.io.vread(video_path)[0:BATCH_SIZE]
        # frames = torch.tensor(frames).float().cuda()
        #0.1 and c = 0.5 work. c=0.54 and 0.07 LR works even better.
        self.learning_rate = 0.07
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
            self.delta = Variable(torch.zeros(frames.shape), requires_grad=True)
        else:
            self.delta = Variable(torch.zeros(frames.shape), requires_grad=True)

        self.optimizer = optim.Adam([self.delta],
                                    lr=self.learning_rate,
                                    betas=(0.9, 0.999))
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.20)
        self.input_shape = (299, 299)
        self.target = target

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

    def loss(self, pass_in):
        tf_img_fn = ptm_utils.TransformImage(self.oracle.conv)
        load_img_fn = PIL.Image.fromarray

        # pass_in want to find seq_preds aka logits
        batches = create_batches(pass_in, tf_img_fn, load_img_fn)
        feats = self.oracle.conv_forward(batches)
        seq_prob, seq_preds = self.oracle.encoder_decoder_forward(feats, mode='inference')

        # caption = []
        # caption.extend([vocab(token) for token in target.split(' ')])
        # caption = torch.Tensor(caption)
        #
        # lengths = [len(cap) for cap in caption]
        # targets = torch.zeros(len(caption), max(lengths)).long()
        # for i, cap in enumerate(caption):
        #     end = lengths[i]
        #     targets[i, :end] = cap[:end]

        crit = utils.LanguageModelCriterion()
        loss = crit(seq_prob.unsqueeze(0), self.tlabel[:, 1:].cuda(), self.tmask[:, 1:].cuda())
        # loss_fn = nn.NLLLoss(reduce=False)
        #
        # logits = to_contiguous(seq_preds).view(-1, seq_preds.shape[2])
        # loss =loss_fn(logits, targets)

        return loss

    # Execute uses the image path directly. Fix out_dir later, for now it's the same directory (add an argument for output dir for argparse)
    def execute(self, video_path):

        tf_img_fn = ptm_utils.TransformImage(self.oracle.conv)
        conv_shape = tf_img_fn.input_size
        load_img_fn = PIL.Image.fromarray

        print(video_path)
        frames = skvideo.io.vread(video_path)[:BATCH_SIZE]
        # plt.imshow(frames[0])
        # plt.show()

        with torch.no_grad():
            # bp ---
            o_batches = o_create_batches(frames, load_img_fn, tf_img_fn)
            seq_prob, seq_preds = self.oracle(o_batches, mode='inference')
            sents = utils.decode_sequence(self.vocab, seq_preds)

            for sent in sents:
                print('Original caption: ' + sent)

        print('Target caption: ' + self.target)

        #all the frames are stored in batches. Batches[0] should contain the first 32 frames.
        torch.cuda.empty_cache()
        #batches = batches.float().cuda()
        # dc = 0.80
        dc = 10.0
        #c is some constant between 0 and 1

        # original = torch.tensor(frames).float().cuda()
        original = torch.tensor(frames)

        #c = 0.5
        c=0.54
        # The attack
        for i in range(self.num_iterations):

            apply_delta = torch.clamp(self.delta, min=-dc, max=dc)

            #The perturbation is applied to the original and resized through interpolation
            # pass_in = original.cuda()
            pass_in = original.float()
            s = pass_in.shape
            #pass_in = torch.nn.functional.interpolate(original, size=self.input_shape, mode='bilinear')
            # pass_in = m_normalize(pass_in.squeeze()).unsqueeze(0)
            pass_in = torch.clamp(apply_delta + pass_in, min=0, max =255)
            # pass_in = torch.nn.functional.interpolate(pass_in.view(s[0], s[3], s[1], s[2]), size=tuple(conv_shape)[1:], mode='bilinear')
            # pass_in = pass_in.view(*pass_in.size())
            # pass_in.to(device)

            cost = self.loss(pass_in)
            #cost calculated with the adversarial image
            #cost = self.oracle.forward(pass_in, self.target)


            #w and y make calculations more efficient and are used to calculate the l2 norm
            y = torch_arctanh(torch.nn.functional.interpolate(pass_in, size=self.input_shape, mode='bilinear'))
            w = torch_arctanh(pass_in) - y
            normterm = (w+y).tanh() - y.tanh()
            cost = c * cost + normterm.norm()

            #calculate gradients
            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()

            #Iteration and cost displayed at every step. We apply the perturbation to the original image again to find the adversarial caption.
            logger.debug("iteration: {}, cost: {}".format(i, cost))
            adv_sample = torch.clamp(apply_delta + frames, min=0.0, max=1.0)
            batches = create_batches(adv_sample)
            seq_prob, seq_preds = self.oracle(batches, mode='inference')
            sents = utils.decode_sequence(self.vocab, seq_preds)

            #print(sents[0])


            #Every iteration it checks for whether or not the target caption equals the original
            if sents[0] == self.target:
                # We're done
                logger.debug("Decoding at iteration {}:  {} ".format(i, sents[0]))
                logger.debug("Early stop. Cost: {}".format(cost))
                plt.imshow(frames[30]/255.0)
                plt.show()
                break

            #Every 10 iterations it outputs the caption.
            if i % 10 == 0:
                # See how we're doing
                logger.debug("Decoding at iteration {}: {} ".format(i, sents[0]))

                if sents[0] == self.target:
                    # We're done
                    logger.debug("Early stop.")
                    plt.imshow(frames[30])
                    plt.show()
                    break

            #Every 500 iterations it outputs an image with the perturbation applied.
            if i % 500 == 0:
                plt.imshow(frames[30])
                plt.show()


        self.oracle.encoder.eval()
        self.oracle.decoder.eval()


        #Once everything is done, it will save the adversarial image by appending _adversarial to the original target file's name and uses its format.
        print(video_path)
        adv_image = self._tensor_to_PIL_im(adv_sample)
        imgpath = video_path.split('/')
        advpath = '' + imgpath[0]
        for i in range(1, len(imgpath)-1):
            advpath += '/%s' % imgpath[i]

        print(advpath)
        filename = imgpath[len(imgpath)-1].split('.')
        advpath += '/%s_adversarial.%s' % (filename[0], filename[1])
        adv_image.save(advpath)
        print(advpath)


def PIL_to_image(image_path):
    tensorize = transforms.ToTensor()
    original_pil = Image.open(image_path)
    image_tensor = tensorize(original_pil).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    image_tensor = Variable(image_tensor)
    return image_tensor



def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5


def create_batches(frames_to_do, tf_img_fn, load_image_fn, batch_size=BATCH_SIZE):
    n = frames_to_do.shape[0]

    if n < batch_size:
        logger.warning("Sample size less than batch size: Cutting batch size.")
        batch_size = n

    logger.info("Generating {} batches...".format(n // batch_size))
    batches = []

    for idx in range (0, n, batch_size):
        frames_idx = list(range(idx, min(idx + batch_size, n)))

        batch_frames = frames_to_do[frames_idx]
        batch_tensor = torch.zeros((len(batch_frames),) + tuple(tf_img_fn.input_size))
        for i, frame_ in enumerate(batch_frames):
            inp = load_image_fn(frame_.detach().numpy().astype(np.uint8))
            input_tensor = tf_img_fn(inp)  # 3x400x225 -> 3x299x299 size may differ
            # input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
            batch_tensor[i] = input_tensor
        # batches.append(frames_to_do[frames_idx])

        batch_ag = torch.autograd.Variable(batch_tensor, requires_grad=False)
        batches.append(batch_ag)

    return batches


def o_create_batches(frames_to_do, load_img_fn, tf_img_fn, batch_size=BATCH_SIZE):
    n = len(frames_to_do)
    if n < batch_size:
        logger.warning("Sample size less than batch size: Cutting batch size.")
        batch_size = n

    logger.info("Generating {} batches...".format(n // batch_size))
    batches = []
    frames_to_do = np.array(frames_to_do)

    for idx in range(0, n, batch_size):
        frames_idx = list(range(idx, min(idx+batch_size, n)))
        batch_frames = frames_to_do[frames_idx]

        batch_tensor = torch.zeros((len(batch_frames),) + tuple(tf_img_fn.input_size))
        for i, frame_ in enumerate(batch_frames):
            input_img = load_img_fn(frame_)
            input_tensor = tf_img_fn(input_img)  # 3x400x225 -> 3x299x299 size may differ
            # input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
            batch_tensor[i] = input_tensor

        batch_ag = torch.autograd.Variable(batch_tensor, requires_grad=False)
        batches.append(batch_ag)

    return batches

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()
