from utils import *

import torch.nn as nn
import torch
import numpy as np
import pretrainedmodels
import json
import os
import argparse
import skvideo.io
import torch
import pickle
import PIL
import numpy as np
from video_caption_pytorch.models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from video_caption_pytorch.dataloader import VideoDataset
from pretrainedmodels import utils as ptm_utils
from video_caption_pytorch.process_features import process_batches, create_batches
# from video_caption_pytorch.models.ConvS2VT import ConvS2VT
from video_caption_pytorch.misc import utils as utils
import math
from Attack_Models.S2VT_Attack import S2VT_Attack
import torch.optim as optim
# from video_attack import CarliniAttack
from torchvision import transforms
from torch.autograd import Variable
from Attack import Attack
from video_caption_pytorch.train import train
from torch.utils.data import DataLoader

''' Most recent configuration as of 10/15/2019
python _unittest_video_attack.py
"C:/Path/To/Directory/InputVideo.avi" 
--recover_opt "C:/Path/To/opt_info.json" 
--saved_model "C:/Path/To/model_1000.pth"
'''

'''

"D:\College\Research\December 2018 Video Captioning Attack\video captioner\YouTubeClips\AJJ-iQkbRNE_97_109.avi" --recover_opt "D:\College\Research\December 2018 Video Captioning Attack\video captioner\save\msvd_nasnetalarge\opt_info.json" --saved_model "D:\College\Research\December 2018 Video Captioning Attack\video captioner\save\msvd_nasnetalarge\model_1000.pth"

'''

# target_caption = '<sos> A man is moving a toy <eos>'
BATCH_SIZE = 3

'''model = nn.Sequential(models.vgg19(True), YourModule())
# Use it as usual
out = model(input)'''

# def train(loader, model, crit, optimizer, lr_scheduler, opt, rl_crit=None):

def main(args, opt):

    testpath = 'D:\\College\\Research\\2019 Video Captioning Attack Conference Paper\\youtube2text_preprocessed_for_arctic_capgen_vid\\youtube2text_iccv15\\dict_movieID_caption.pkl'

    with open(testpath, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    print(data)


    dataset = VideoDataset(opt, 'inference')
    opt["vocab_size"] = dataset.get_vocab_size()
    opt["seq_length"] = dataset.max_len

    if opt['beam_size'] != 1:
        assert opt["batch_size"] == 1
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(opt["vocab_size"], opt["max_len"], opt["dim_hidden"], opt["dim_word"], opt['dim_vid'],
                          n_layers=opt['num_layers'],
                          rnn_cell=opt['rnn_type'],
                          bidirectional=opt["bidirectional"],
                          rnn_dropout_p=opt["rnn_dropout_p"])
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(opt["dim_vid"], opt["dim_hidden"],
                             n_layers=opt['num_layers'],
                             rnn_cell=opt['rnn_type'], bidirectional=opt["bidirectional"],
                             input_dropout_p=opt["input_dropout_p"], rnn_dropout_p=opt["rnn_dropout_p"])
        decoder = DecoderRNN(opt["vocab_size"], opt["max_len"], opt["dim_hidden"], opt["dim_word"],
                             n_layers=opt['num_layers'],
                             rnn_cell=opt['rnn_type'], input_dropout_p=opt["input_dropout_p"],
                             rnn_dropout_p=opt["rnn_dropout_p"], bidirectional=opt["bidirectional"])
        model = S2VTAttModel(encoder, decoder)
    else:
        return

    # if torch.cuda.device_count() > 1:
    #     print("{} devices detected, switch to parallel model.".format(torch.cuda.device_count()))
    #     model = nn.DataParallel(model)

    #model, videopath, targetcap, dataset, config, optimizer, crit, window

    #config: batch_size, c, learning rate, num it,input shape

    config = {

        "batch_size": BATCH_SIZE,
        "c": 100,
        "learning_rate": 0.005,
        "num_iterations": 1000,
        "input_shape": (299, 299),
        "num_frames": 288,
        "dimensions": 331

    }

    convnet = 'nasnetalarge'
    full_decoder = ConvS2VT(convnet, model, opt)

    # model = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, padding=0, stride=2,
    #                                             bias=False), full_decoder)

    #loader, model, crit, optimizer, lr_scheduler, opt, rl_crit=None

    dataset = VideoDataset(opt, 'train')
    dataloader = DataLoader(dataset,
                            batch_size=opt["batch_size"],
                            num_workers=16,
                            shuffle=True)

    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"])
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt["learning_rate_decay_every"],
        gamma=opt["learning_rate_decay_rate"])

    train(dataloader, model, crit, optimizer, exp_lr_scheduler, opt, rl_crit)

    #
# carlini = CarliniAttack(oracle=full_decoder, video_path=video_path, target=target_caption, dataset=dataset)

    # carlini.execute(video_path)



class ConvS2VT(nn.Module):
    def __init__(self, conv_name, s2vt, opt):
        """
        A full Conv + S2VT model pipeline
        :param conv: The FC feature extractor
        :param s2vt: Complete S2VT* model
        """
        super(ConvS2VT, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv_name = conv_name
        self.conv = pretrainedmodels.__dict__[conv_name](num_classes=1000, pretrained='imagenet')
        self.conv.eval()
        self.conv.to(self.device)

        self.s2vt = s2vt.to(self.device)
        # self.s2vt.load_state_dict(torch.load(opt["saved_model"]))

    def forward(self, frame_batches, target_variable=None, mode='train', get_attn=False, opt={}, single_batch=True):
        """

        Args:
            target_variable (None, optional): ground truth labels

        Returns:
            seq_prob: Variable of shape [batch_size, max_len-1, vocab_size]
            seq_preds: [] or Variable of shape [batch_size, max_len-1]
        """
        vid_feats = process_batches(frame_batches, self.conv_name, [0], self.conv, single_batch=single_batch)
        vid_feats = vid_feats.unsqueeze(0)
        if (get_attn == True):
            attn = self.s2vt(vid_feats, mode=mode, get_attn=get_attn, opt=opt)
            return attn

        # TODO: Batch n videos and feed
        seq_probs, seq_preds = self.s2vt(vid_feats, mode=mode, opt=opt)
        return seq_probs, seq_preds

    def conv_forward(self, frame_batches):
        feats = process_batches(frame_batches, self.conv_name, [0], self.conv)
        # feats = np.array([feats])
        feats = feats.unsqueeze(0)

        return feats

    def encoder_decoder_forward(self, vid_feats, target_variable=None, mode='train', get_attn=False, opt={}):
        # vid_feats = torch.from_numpy(vid_feats).to(self.device)
        if(get_attn == True):
            attn = self.s2vt(vid_feats, get_attn=get_attn, mode=mode, opt=opt, target_variable=target_variable)
            return attn

        seq_probs, seq_preds = self.s2vt(vid_feats, get_attn=get_attn, mode=mode, opt=opt, target_variable=target_variable)
        return seq_probs, seq_preds




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


def exp_create_batches(frames_to_do, batch_size=BATCH_SIZE):
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
        # logger.warning("Sample size less than batch size: Cutting batch size.")
        batch_size = n

    # logger.info("Generating {} batches...".format(n // batch_size))

    for idx in range (0, n, batch_size):
        frames_idx = list(range(idx, min(idx + batch_size, n)))

        # <batch, h, w, ch> <0,255>
        batch_tensor = Variable(torch.tensor(frames_to_do[frames_idx]).float())

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('videos', nargs='+',
                        help='space delimited list of video paths')
    parser.add_argument('--recover_opt', type=str, required=True,
                        help='recover train opts from saved opt_json')
    parser.add_argument('--saved_model', type=str, default='',
                        help='path to saved model to evaluate')
    # parser.add_argument('--rnn_type', type=str, default='gru', help='lstm or gru')
    parser.add_argument('--dump_json', type=int, default=1,
                        help='Dump json with predictions into vis folder? (1=yes,0=no)')
    parser.add_argument('--results_path', type=str, default='results/')
    parser.add_argument('--dump_path', type=int, default=0,
                        help='Write image paths along with predictions into vis json? (1=yes,0=no)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu device number')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--sample_max', type=int, default=1,
                        help='0/1. whether sample max probs  to get next word in inference stage')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--beam_size', type=int, default=1,
                        help='used when sample_max = 1. Usually 2 or 3 works well.')

    args = parser.parse_args()
    args = vars((args))
    opt = json.load(open(args["recover_opt"]))
    for k, v in args.items():
        opt[k] = v
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]



    main(args, opt)