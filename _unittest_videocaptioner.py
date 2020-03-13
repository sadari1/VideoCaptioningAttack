import argparse
import torch

from utils import *
from ImageCaptioner import ImageCaptioner
from yunjey_image_captioning.build_vocab import Vocabulary
from torchvision import transforms

from image_attack import CarliniAttack

import json
import os
import argparse
import skvideo.io
import torch
import numpy as np
import PIL
from video_caption_pytorch.models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from video_caption_pytorch.dataloader import VideoDataset
from pretrainedmodels import utils as ptm_utils
from video_caption_pytorch.process_features import process_batches, create_batches
from video_caption_pytorch.models.ConvS2VT import ConvS2VT
from video_caption_pytorch.misc import utils as utils
import matplotlib.pyplot as plt

#from video_attack import CarliniAttack


'''
python _unittest_videocaptioner.py
"C:/Path/To/Video.avi" 
--recover_opt "C:/Path/To/opt_info.json" 
--saved_model "C:/Path/To/model_1000.pth"
'''


'''

"D:\College\Research\December 2018 Video Captioning Attack\video captioner\YouTubeClips\AJJ-iQkbRNE_97_109.avi" --recover_opt "D:\College\Research\December 2018 Video Captioning Attack\video captioner\save\msvd_nasnetalarge\opt_info.json" --saved_model "D:\College\Research\December 2018 Video Captioning Attack\video captioner\save\msvd_nasnetalarge\model_1000.pth"

'''

import torch
torch.manual_seed(117)
import numpy
numpy.random.seed(117)

def main(opt):
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

    # convnet = 'nasnetalarge'
    # convnet = 'resnet152'
    convnet = 'vgg16'
    vocab = dataset.get_vocab()
    full_decoder = ConvS2VT(convnet, model, opt)
#D:\College\Research\December 2018 Video Captioning Attack\video captioner\YouTubeClips\CNN_SaYwh6chmiw_15_40.npy
    videos = {

        # 1: 'RSx5G0_xH48_12_17.avi',
        2: 'nc8hwLaOyZU_1_19_adversarialWINDOW.avi',
        3: 'O2qiPS2NCeY_2_18_adversarialWINDOW.avi',
        4: 'kI6MWZrl8v8_149_161_adversarialWINDOW.avi',
        5: 'X7sQq-Iu1gQ_12_22_adversarialWINDOW.avi',
        6: '77iDIp40m9E_159_181_adversarialWINDOW.avi',
        7: 'SaYwh6chmiw_15_40_adversarialWINDOW.avi',
        8: 'pFSoWsocv0g_8_17_adversarialWINDOW.avi',
        9: 'HmVPxs4ygMc_44_53_adversarialWINDOW.avi',
        10: 'glii-kazad8_21_29_adversarialWINDOW.avi',
        11: 'AJJ-iQkbRNE_97_109_adversarialWINDOW.avi'

    }

    #video_path = 'D:\\College\Research\\December 2018 Video Captioning Attack\\video captioner\\YouTubeClips\\ACOmKiJDkA4_49_54.avi'
    # video_path = opt['videos'][0]

    # video_path = 'D:\\College\\Research\\December 2018 Video Captioning Attack\\video captioner\\YouTubeClips\\' + \
    #              videos[2]

    video_path = 'D:\\College\\Research\\December 2018 Video Captioning Attack\\video captioner\\YouTubeClips\\vgg16Adversarial_SaYwh6chmiw_15_40.avi'

    numpy_path = "D:\\College\\Research\\December 2018 Video Captioning Attack\\video captioner\\YouTubeClips\\vgg16CNN_SaYwh6chmiw_15_40.npy"
    adv_frames = np.load(numpy_path)


    tf_img_fn = ptm_utils.TransformImage(full_decoder.conv)
    load_img_fn = PIL.Image.fromarray

    print(video_path)
    with torch.no_grad():
        frames = skvideo.io.vread(video_path)
        print("Total frames: {}".format(len(frames)))
        # print(frames[[0, 1, 2, 3, 4, 5]].shape)
        plt.imshow(frames[0]*255.)
        plt.show()


        # bp ---
        batches = create_batches(frames, load_img_fn, tf_img_fn)
        seq_prob, seq_preds = full_decoder(batches, mode='inference')
        sents = utils.decode_sequence(vocab, seq_preds)

        print(sents[0])


        np_frames = adv_frames.astype(np.uint8)
        print("Numpy CNN frames \nTotal frames: {}".format(len(np_frames)))
        # print(frames[[0, 1, 2, 3, 4, 5]].shape)
        plt.imshow(np_frames[0]*255.)
        plt.show()

        # bp ---
        batches = create_batches(np_frames, load_img_fn, tf_img_fn)
        seq_prob, seq_preds = full_decoder(batches, mode='inference')
        sents = utils.decode_sequence(vocab, seq_preds)

        print(sents[0])


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
    main(opt)