from utils import *

import json
import os
import argparse
import skvideo.io
import torch
import PIL
import numpy as np
import matplotlib.pyplot as plt
from video_caption_pytorch.models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from video_caption_pytorch.dataloader import VideoDataset
from pretrainedmodels import utils as ptm_utils
from video_caption_pytorch.process_features import process_batches, create_batches
from video_caption_pytorch.models.ConvS2VT import ConvS2VT
from video_caption_pytorch.misc import utils as utils

from video_attack import CarliniAttack

''' Most recent configuration as of 10/15/2019
python _unittest_video_attack.py
"C:/Path/To/Directory/InputVideo.avi" 
--recover_opt "C:/Path/To/opt_info.json" 
--saved_model "C:/Path/To/model_1000.pth"
'''


'''

"D:\College\Research\December 2018 Video Captioning Attack\video captioner\YouTubeClips\AJJ-iQkbRNE_97_109.avi" --recover_opt "D:\College\Research\December 2018 Video Captioning Attack\video captioner\save\msvd_nasnetalarge\opt_info.json" --saved_model "D:\College\Research\December 2018 Video Captioning Attack\video captioner\save\msvd_nasnetalarge\model_1000.pth"

'''

target_caption = '<sos> A man is moving a toy <eos>'
ATTACK_BATCH_SIZE = 3
BATCH_SIZE = 32


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



    convnet = 'nasnetalarge'
    full_decoder = ConvS2VT(convnet, model, opt)

    #'A woman is cutting a green onion'
    video_path = opt['videos'][0]

    tf_img_fn = ptm_utils.TransformImage(full_decoder.conv)
    load_img_fn = PIL.Image.fromarray
    vocab = dataset.get_vocab()

    length = len(skvideo.io.vread(video_path)) / 8
    print("Total number of frames: {}".format(len(skvideo.io.vread(video_path))))
    print("Total number of frames to do: {}".format(length))

    with torch.no_grad():
        frames = skvideo.io.vread(video_path)

        # bp ---

        attn_weights = []

        total_iterations = np.ceil(length / BATCH_SIZE)
        iteration = 1
        frame_counter = 0

        while(frame_counter < length):
            if length - frame_counter < BATCH_SIZE:
                batches = create_batches(frames[frame_counter:int(length)], load_img_fn, tf_img_fn)
                attn = full_decoder(batches, mode='inference', get_attn=True)
                frame_counter = frame_counter + (length - frame_counter)
            else:
                batches = create_batches(frames[frame_counter:frame_counter+BATCH_SIZE-1],load_img_fn, tf_img_fn)
                attn = full_decoder(batches, mode='inference', get_attn=True)
                frame_counter = frame_counter + BATCH_SIZE
            # print(attn.shape, attn[0].shape, type(attn))

            attn = attn.cpu().detach().numpy().tolist()[0]

            print("Weights for batch {}: {}".format(iteration, attn))
            for f in attn:
                attn_weights.append(f)
            iteration = iteration + 1

            # attn_weights.append(attn.cpu().detach().numpy().tolist()[0])



        batches = create_batches(frames, load_img_fn, tf_img_fn)
        seq_prob, seq_preds = full_decoder(batches, mode='inference', get_attn = False)
        sents = utils.decode_sequence(vocab, seq_preds)

        original_caption = sents[0]

    print(attn_weights)

    att_window = np.sort(np.argpartition(attn_weights, -ATTACK_BATCH_SIZE)[-ATTACK_BATCH_SIZE:]).tolist()


    print("Indices of frames with highest attention weights: {}".format(att_window))
    #video_path = 'D:\\College\Research\\December 2018 Video Captioning Attack\\video captioner\\YouTubeClips\\ACOmKiJDkA4_49_54.avi'
    # target_caption = '<sos> A man is moving a toy <eos>'
    # target_caption = '<sos> A boy is kicking a soccer ball into the goal <eos>'


    adv_frames = []
    carlini = CarliniAttack(oracle=full_decoder, video_path=video_path, target=target_caption, dataset=dataset,
                                                            att_window=att_window)
    finished_frames = carlini.execute(video_path, att_window=att_window, functional=True)
    adv_frames.append(finished_frames.detach().cpu().numpy())



    base_toks = video_path.split('/')
    base_dir_toks = base_toks[:-1]
    base_filename = base_toks[-1]
    base_name = ''.join(base_filename.split('.')[:-1])
    adv_path = os.path.join('/'.join(base_dir_toks), base_name + '_adversarial.avi')

    print("\nSaving to: {}".format(adv_path))
    adv_frames = np.concatenate(adv_frames, axis=0)
    outputfile = adv_path
    writer = skvideo.io.FFmpegWriter(outputfile, outputdict={
        '-vcodec': 'libx264',  # use the h.264 codec
        '-crf': '0',  # set the constant rate factor to 0, which is lossless
        '-vb': '50M',
        '-r': '25',
        '-preset': 'ultrafast'  # the slower the better compression, in princple, try
    })
    for f in adv_frames:
        writer.writeFrame(f)
    print(len(adv_frames))

    # skvideo.io.vwrite(adv_path, adv_frames)
    writer.close()

    with torch.no_grad():
        a_frames = skvideo.io.vread(adv_path)

        # frames = skvideo.io.vread(video_path)

        # for f in range(0, len(att_window)):
        #     frames[att_window[f]] = a_frames[f]

        # frames = frames[:50]
        # frames = adv_frames
        # print(frames[[0, 1, 2, 3, 4, 5]].shape)
        # plt.imshow(frames[0])
        # plt.show()
        #
        # plt.imshow(adv_frames[0]/255.)
        # plt.show()

        # bp ---


        batches = create_batches(a_frames, load_img_fn, tf_img_fn)
        seq_prob, seq_preds = full_decoder(batches, mode='inference')
        sents = utils.decode_sequence(vocab, seq_preds)

        adv_caption = sents[0]


    print("\nOriginal Caption: {}\nTarget Caption: {}\nAdversarial Caption: {}".format(original_caption, target_caption, adv_caption))

# carlini = CarliniAttack(oracle=full_decoder, video_path=video_path, target=target_caption, dataset=dataset)

    # carlini.execute(video_path)


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