import os
import numpy as np
import torch
import logging
import skvideo.io
import matplotlib.pyplot as plt
from video_caption_pytorch.misc import utils as utils
from torch.autograd import Variable
from torchvision import transforms
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig()
logger = logging.getLogger(__name__)
# Change logging level to info if running experiment, debug otherwise
logger.setLevel(logging.DEBUG)


'''
pass in an object (instance of a class):

Ex: S2VT_Attack:
learning rate
num it
batch size
oracle 
dataset
phrase length is target
vocab

optimizer
crit
input_shape

target
real target
tlabel, tmask

loss function

create batches

'''

#attack package is the attack object for whatever is passed in.
#It is called oracle because it holds the information for basically everything that's needed
class Attack:
    def __init__(self, attack_package):
        """
        :param oracle: ImageCaptioner class
        :param image: sample image to get shape
        :param target: target caption (for now,
        in future target can also be the image we want to extract target caption from)
        """

        self.oracle = attack_package
    '''
    video path, windows and such will stay.
    
    create batches will be part of the class function that you pass in. Attack algorithm otherwise remains the same.
    
    '''

    '''
    The attack: original component, delta (mask), and decode/test phase to calculate gradients
    
    
    '''
    # Execute uses the image path directly. Fix out_dir later, for now it's the same directory (add an argument for output dir for argparse)
    def execute(self, functional=False, stats=False):


        c = self.oracle.c

        attack_algorithm = self.oracle.attack_algorithm
        # plt.imshow(frames[0])
        # plt.show()

        #
        # original = torch.tensor(self.oracle.frames)
        # original = (original.float()).cuda()

        original = self.oracle.original()

        #Original caption/classification, make this a function as well
        # with torch.no_grad():
        #     # bp ---
        #     batch = self.oracle.create_batches(frames_to_do=original)
        #     seq_prob, seq_preds = self.oracle.model(batch.unsqueeze(0), mode='inference')
        #     sents = utils.decode_sequence(self.oracle.vocab, seq_preds)
        #
        #     for sent in sents:
        #         print('Original caption: ' + sent)


        # all the frames are stored in batches. Batches[0] should contain the first 32 frames.
        torch.cuda.empty_cache()

        # The attack
        for i in range(self.oracle.num_iterations):


            print("Video path: ", self.oracle.video_path)

            #Make this a part of the attack model.

            cost, decoded, apply_delta, pass_in = self.oracle.costs_decoded(original, attack_algorithm)
            logger.info("Decoding at iteration {}: {} ".format(i, decoded))




            #Cost, decoded = self.oracle(
            #Instead of sents, maybe it's decoded (so action label becomes target label for example)
            if decoded == self.oracle.real_target or i == (self.oracle.num_iterations - 1):

                # We're done
                logger.debug("Decoding at iteration {}:\t{} ".format(i, decoded))
                logger.debug("Early stop. Cost: {}".format(cost))

                base_toks = self.oracle.video_path.split('/')
                base_dir_toks = base_toks[:-1]
                base_filename = base_toks[-1]
                base_name = ''.join(base_filename.split('.')[:-1])
                adv_path = os.path.join('/'.join(base_dir_toks), base_name + '_adversarial.avi')

                # plt_tensor(pass_in/255.)
                if not functional:
                    self.oracle.save_to(adv_path, apply_delta, pass_in)

                if stats:
                    return {'pass_in': pass_in.detach().cpu().numpy(),
                            'iterations': i,
                            'delta': self.oracle.delta.detach().cpu().numpy()}
                else:
                    return pass_in

            # Every 10 iterations it outputs the caption.
            if i % 1000 == 0:
                # See how we're doing
                logger.info("Decoding at iteration {}: {} ".format(i, decoded))
                if not functional:
                    self.oracle.plt_collate_batch(pass_in / 255.)

            if i % 50 == 0:
                plt_tensor(pass_in/255.) if attack_algorithm == 'carliniwagner' else plt_tensor(pass_in/255.)

            # print("Norm:\t{}\t\tCost:\t{}".format(apply_delta.norm(), cost.data))


            #Since pass_in and original are already acquired based on the attack model,
            #This can be left as is.

            # w and y make calculations more efficient and are used to calculate the l2 norm

            if attack_algorithm == 'carliniwagner':


                # normterm = 0.5 * ((pass_in / 255.).tanh() + 1) - (original / 255.)
                normterm = (pass_in/255.) - (original/255.)
                # Then take the l2 norm of the mean difference
                normterm = normterm.mean(0).norm()

                print(normterm)
                # normterm = normterm.pow(2)

                cost = normterm + c * cost

            else:

                y = torch_arctanh(original / 255.).cuda()
                w = torch_arctanh(pass_in / 255.) - y
                normterm = ((w+y).tanh() - y.tanh())
                normterm = normterm.mean(0).norm()
                print("Cost:\t{}\t+\tNormterm:\t{}".format(cost, normterm))
                cost = cost + (c * normterm)

            # calculate gradients
            self.oracle.optimizer.zero_grad()
            cost.backward()
            self.oracle.optimizer.step()

            # Iteration and cost displayed at every step. We apply the perturbation to the original image again to find the adversarial caption.
            logger.debug("\nIteration: {}, cost: {}".format(i, cost))
            # torch.cuda.empty_cache()
            # Every iteration it checks for whether or not the target caption equals the original

#
# def PIL_to_image(image_path):
#     tensorize = transforms.ToTensor()
#     original_pil = Image.open(image_path)
#     image_tensor = tensorize(original_pil).unsqueeze(0)
#     image_tensor = image_tensor.to(device)
#     image_tensor = Variable(image_tensor)
#     return image_tensor
def plt_tensor(batched_t):
    showing = batched_t[0]
    if batched_t.shape[-1] == 3:
        showing = showing.detach().cpu().numpy()
    else:
        showing = showing.permute(1, 2, 0).detach().cpu().numpy()

    plt.imshow(showing)
    plt.show()



def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5

#
# def plt_tensor(batched_t):
#     showing = batched_t[0]
#     if batched_t.shape[-1] == 3:
#         showing = showing.detach().cpu().numpy()
#     else:
#         showing = showing.permute(1, 2, 0).detach().cpu().numpy()
#
#     plt.imshow(showing)
#     plt.show()


