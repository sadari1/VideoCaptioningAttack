import numpy as np
import torch
import torch.optim as optim
import logging
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from torchvision import transforms
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig()
logger = logging.getLogger(__name__)
# Change logging level to info if running experiment, debug otherwise
logger.setLevel(logging.DEBUG)

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

        # Variable for adversarial noise, which is added to the image to perturb it
        # Starts as an empty mask so noise will be added onto it
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

        # Split up transforms to mimic real attack steps
        tensorize = transforms.ToTensor()
        m_normalize = transforms.Normalize((0.485, 0.456, 0.406),
                                           (0.229, 0.224, 0.225))
        original_i = load_image(image_path, reshape=False)
        #original_i.show()
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
        #c is some constant between 0 and 1

        #c = 0.5
        c=0.54
        # The attack
        for i in range(self.num_iterations):

            apply_delta = torch.clamp(self.delta, min=-dc, max=dc)

            #The perturbation is applied to the original and resized through interpolation
            pass_in = torch.clamp(apply_delta + original, min=0.0, max=1.0)
            pass_in = torch.nn.functional.interpolate(pass_in, size=self.input_shape, mode='bilinear')
            # pass_in = m_normalize(pass_in.squeeze()).unsqueeze(0)
            pass_in = pass_in.view(*pass_in.size())
            pass_in.to(device)

            #cost calculated with the adversarial image
            cost = self.oracle.forward(pass_in, self.target)

            #w and y make calculations more efficient and are used to calculate the l2 norm
            y = torch_arctanh(torch.nn.functional.interpolate(original, size=self.input_shape, mode='bilinear'))
            w = torch_arctanh(pass_in) - y
            normterm = (w+y).tanh() - y.tanh()
            cost = c * cost + normterm.norm()

            #calculate gradients
            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()


            #Iteration and cost displayed at every step. We apply the perturbation to the original image again to find the adversarial caption.
            logger.debug("iteration: {}, cost: {}".format(i, cost))
            adv_sample = torch.clamp(apply_delta + original, min=0.0, max=1.0)
            sim_pred = self.decode_logits(pass_in)

            #Every iteration it checks for whether or not the target caption equals the original
            if sim_pred == self.target:
                # We're done
                logger.debug("Decoding at iteration {}: <start> {} <end>".format(i, sim_pred))
                logger.debug("Early stop. Cost: {}".format(cost))
                self._tensor_to_PIL_im(adv_sample).show()
                break

            #Every 10 iterations it outputs the caption.
            if i % 10 == 0:
                # See how we're doing
                logger.debug("Decoding at iteration {}: <start> {} <end>".format(i, sim_pred))

                if sim_pred == self.target:
                    # We're done
                    logger.debug("Early stop.")
                    self._tensor_to_PIL_im(adv_sample).show()
                    break

            #Every 500 iterations it outputs an image with the perturbation applied.
            if i % 500 == 0:
                self._tensor_to_PIL_im(adv_sample).show()


        self.oracle.encoder.eval()
        self.oracle.decoder.eval()


        #Once everything is done, it will save the adversarial image by appending _adversarial to the original target file's name and uses its format.
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
