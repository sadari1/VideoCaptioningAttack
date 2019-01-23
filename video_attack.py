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

class CarliniAttack:
    def __init__(self, oracle, video_path, target):
        """
        :param oracle: ImageCaptioner class
        :param image: sample image to get shape
        :param target: target caption (for now,
        in future target can also be the image we want to extract target caption from)
        """

        frames = skvideo.io.vread(video_path)[0:32]
        frames = torch.tensor(frames).float().cuda()
        #0.1 and c = 0.5 work. c=0.54 and 0.07 LR works even better.
        self.learning_rate = 0.07
        # self.learning_rate = 10
        self.num_iterations = 50000
        # self.num_iterations = 100
        self.batch_size = 1
        self.phrase_length = len(target)
        self.oracle = oracle

        # Variable for adversarial noise, which is added to the image to perturb it
        # Starts as an empty mask so noise will be added onto it
        if torch.cuda.is_available():
            self.delta = Variable(torch.zeros(frames.shape).cuda(), requires_grad=True)
        else:
            self.delta = Variable(torch.zeros(frames.shape), requires_grad=True)

        self.optimizer = optim.Adam([self.delta],
                                    lr=self.learning_rate,
                                    betas=(0.9, 0.999))
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.20)
        self.input_shape = (299, 299)
        self.target = target


    # Execute uses the image path directly. Fix out_dir later, for now it's the same directory (add an argument for output dir for argparse)
    def execute(self, video_path, vocab):

        tf_img_fn = ptm_utils.TransformImage(self.oracle.conv)
        #load_img_fn = PIL.Image.fromarray

        print(video_path)
        with torch.no_grad():
            frames = skvideo.io.vread(video_path)
            frames = frames.astype(np.float32)
            frames = torch.tensor(frames).cuda()
            plt.imshow(frames[30]/255.0)
            plt.show()

            # bp ---
            batches = create_batches(frames, tf_img_fn)
            seq_prob, seq_preds = self.oracle(batches, mode='inference')
            sents = utils.decode_sequence(vocab, seq_preds)

            for sent in sents:
                print('Original caption: ' + sent)
        print('Target caption: ' + self.target)

        # dc = 0.80
        dc = 1.0
        #c is some constant between 0 and 1

        #c = 0.5
        c=0.54
        # The attack
        for i in range(self.num_iterations):

            apply_delta = torch.clamp(self.delta, min=-dc, max=dc)

            #The perturbation is applied to the original and resized through interpolation

            pass_in = Apply_Delta(apply_delta, frames, self.input_shape)
            pass_in.to(device)

            #cost calculated with the adversarial image
            cost = self.oracle.forward(pass_in, self.target)


            #w and y make calculations more efficient and are used to calculate the l2 norm
            y = torch_arctanh(torch.nn.functional.interpolate(frames, size=self.input_shape, mode='bilinear'))
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
            batches = create_batches(adv_sample, tf_img_fn)
            seq_prob, seq_preds = self.oracle(batches, mode='inference')
            sents = utils.decode_sequence(vocab, seq_preds)

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
                self._tensor_to_PIL_im(adv_sample).show()


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

def Apply_Delta(delta, original, input_shape):
    pass_in = original
    pass_in[0:32] = torch.clamp(delta + original[0:32], min=0.0, max=1.0)
    pass_in = torch.nn.functional.interpolate(pass_in, size=input_shape, mode='bilinear')
    # pass_in = m_normalize(pass_in.squeeze()).unsqueeze(0)
    pass_in = pass_in.view(*pass_in.size())
    return pass_in

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


def process_batches(batches, ftype, gpu_list, model):
    done_batches = []
    for i, batch in enumerate(batches):
        if torch.cuda.is_available():
            batch = batch.cuda(device=gpu_list[0])

        output_features = model.features(batch)
        output_features = output_features.data.cpu()

        conv_size = output_features.shape[-1]

        if ftype == 'nasnetalarge' or ftype == 'pnasnet5large':
            relu = nn.ReLU()
            rf = relu(output_features)
            avg_pool = nn.AvgPool2d(conv_size, stride=1, padding=0)
            out_feats = avg_pool(rf)
        else:
            avg_pool = nn.AvgPool2d(conv_size, stride=1, padding=0)
            out_feats = avg_pool(output_features)

        out_feats = out_feats.view(out_feats.size(0), -1)
        logger.info('Processed {}/{} batches.\r'.format(i + 1, len(batches)))

        done_batches.append(out_feats)
    feats = np.concatenate(done_batches, axis=0)
    return feats


def create_batches(frames_to_do, tf_img_fn, batch_size=32):
    n = frames_to_do.shape[0]
    if n < batch_size:
        logger.warning("Sample size less than batch size: Cutting batch size.")
        batch_size = n

    logger.info("Generating {} batches...".format(n // batch_size))
    batches = []
    #frames_to_do = np.array(frames_to_do)

    for idx in range(0, n, batch_size):
        frames_idx = list(range(idx, min(idx+batch_size, n)))
        batch_frames = frames_to_do[frames_idx]

        batch_tensor = torch.zeros((batch_frames.shape[0],) + tuple(tf_img_fn.input_size))
        #for i, frame_ in enumerate(batch_frames):
            #input_img = load_img_fn(frame_)
            #input_tensor = tf_img_fn(input_img)  # 3x400x225 -> 3x299x299 size may differ
            # input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299

        input_tensor = torch.nn.functional.interpolate(batch_tensor, size=(299, 299), mode='bilinear')
            #batch_tensor[i] = input_tensor

        #batch_ag = torch.autograd.Variable(batch_tensor, requires_grad=False)
        batches.append(input_tensor)

    return batches
