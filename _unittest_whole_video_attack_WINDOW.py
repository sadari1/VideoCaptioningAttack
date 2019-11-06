from utils import *

import json
import os
import argparse
import skvideo.io
import torch
import PIL
import numpy as np
import cv2
import matplotlib.pyplot as plt
from video_caption_pytorch.models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from video_caption_pytorch.dataloader import VideoDataset
from pretrainedmodels import utils as ptm_utils
from video_caption_pytorch.process_features import process_batches, create_batches
from video_caption_pytorch.models.ConvS2VT import ConvS2VT
from video_caption_pytorch.misc import utils as utils
import math
from video_attack import CarliniAttack
from torchvision import transforms
from torch.autograd import Variable
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

    vid_id = video_path.split('/')[-1]
    vid_id = vid_id.split('.')[0]

    viable_ids = dataset.splits['test'] + dataset.splits['val']
    viable_target_captions = []
    for v_id in viable_ids:
        if v_id == vid_id:
            continue
        plausible_caps = [' '.join(toks) for toks in dataset.vid_to_meta[v_id]['final_captions']]
        viable_target_captions.extend(plausible_caps)

    #Random target caption
    # target_caption = np.random.choice(viable_target_captions)
    # target_caption = '<sos> A man is moving a toy <eos>'
    target_caption = '<sos> A boy is kicking a soccer ball into the goal <eos>'




    # frames = skvideo.io.vread('D:\\College\Research\\December 2018 Video Captioning Attack\\video captioner\\YouTubeClips\\AJJ-iQkbRNE_97_109_adversarialWINDOW.avi', outputdict=
    # {
    #     # '-c:v': 'huffyuv'
    #     '-c:v': 'libx264',
    #     '-crf': '0',
    #     '-preset': 'ultrafast'
    # })
    #
    # # bp ---
    # batches = create_batches(frames, load_img_fn, tf_img_fn)
    # # batches = exp_create_batches(frames, 3)
    # # feats = full_decoder.conv_forward((batches.unsqueeze(0)))
    # # seq_prob, seq_preds = full_decoder.encoder_decoder_forward(feats, mode='inference')
    # seq_prob, seq_preds = full_decoder(batches, mode='inference')
    # sents = utils.decode_sequence(vocab, seq_preds)
    #
    # print(sents[0])




    with torch.no_grad():
        frames = skvideo.io.vread(video_path)

        # bp ---
        batches = create_batches(frames, load_img_fn, tf_img_fn)
        seq_prob, seq_preds = full_decoder(batches, mode='inference')
        sents = utils.decode_sequence(vocab, seq_preds)

        original_caption = sents[0]

    #video_path = 'D:\\College\Research\\December 2018 Video Captioning Attack\\video captioner\\YouTubeClips\\ACOmKiJDkA4_49_54.avi'
    # target_caption = '<sos> A man is moving a toy <eos>'
    # target_caption = '<sos> A boy is kicking a soccer ball into the goal <eos>'


    length = len(skvideo.io.vread(video_path))/96

    print("Total number of frames: {}".format(length))
    adv_frames = []
    iteration = 1
    frame_counter = 0

    total_iterations = np.ceil(length / BATCH_SIZE)
    while(frame_counter < length):
        print("\n\n\nIteration {}/{}".format(iteration, int(total_iterations)))
        iteration = iteration + 1
        if length - frame_counter < BATCH_SIZE:
            window = [frame_counter, length]
            frame_counter = frame_counter + (length - frame_counter)
            print("Using frames {}".format(window))
            print("Frame counter at: {}\nTotal length is: {}\n".format(frame_counter, length))
            carlini = CarliniAttack(oracle=full_decoder, video_path=video_path, target=target_caption, dataset=dataset,
                                    window=window)
            finished_frames = carlini.execute(video_path, window=window, functional=True)
            adv_frames.append(finished_frames.detach().cpu().numpy())

        else:
            window = [frame_counter, frame_counter + BATCH_SIZE-1]
            print("Using frames {}".format(window))
            print("Frame counter at: {}\nTotal length is: {}\n".format(frame_counter, length))
            carlini = CarliniAttack(oracle=full_decoder, video_path=video_path, target=target_caption, dataset=dataset, window = window)
            finished_frames = carlini.execute(video_path, window=window, functional=True)
            adv_frames.append(finished_frames.detach().cpu().numpy())
            frame_counter = frame_counter + BATCH_SIZE

    base_toks = video_path.split('/')
    base_dir_toks = base_toks[:-1]
    base_filename = base_toks[-1]
    base_name = ''.join(base_filename.split('.')[:-1])
    adv_path = os.path.join('/'.join(base_dir_toks), base_name + '_adversarialWINDOW.avi')

    print("\nSaving to: {}".format(adv_path))
    adv_frames = np.concatenate(adv_frames, axis=0)
    # skvideo.io.vwrite(adv_path, adv_frames)

    print("Dimensions: {}".format(adv_frames.shape))
    outputfile = adv_path



    writer = skvideo.io.FFmpegWriter(outputfile, outputdict={
        #huffyuv is lossless

        # '-c:v': 'libx264', #libx264 # use the h.264 codec
        '-c:v': 'r10k', #r210 huffyuv rawvideo ffv1 r10k V408
        # '-colorspace': '0',
        '-pix_fmt': 'rgb32',
        '-b': '50M'
        # '-crf': '0', # set the constant rate factor to 0, which is lossless
        # '-vb': '50M',
        # '-q:v': '1',
        # '-f': 'rawvideo',
        # '-r': '25',
        # '-vpre': 'max',
        # '-preset': 'ultrafast'  # ultrafast, veryslow the slower the better compression, in princple, try
    })
    for f in adv_frames:
        writer.writeFrame(f)

    # skvideo.io.vwrite(adv_path, adv_frames)
    writer.close()

    #ffv1 0.215807946043995
    #huffyuv 0.21578424050191813
    #libx264 0.2341074901578537
    #r210 -0.7831487262059795, -0.7833399258537526
    #gif 0.6889478809555243
    #png 0.2158991440582696 0.21616862708842177
    #qtrle  0.21581286337807626
    #flashsv 0.21610510459932186 0.21600030673323545
    #ffvhuff 0.21620682250167533
    #r10k
    #rawvideo 0.21595001
    with torch.no_grad():

        frames = skvideo.io.vread(adv_path)
        #
        # frames = skvideo.io.vread(adv_path,  inputdict=
        # {
        #     # '-pix_fmt': 'argb'
        #     # '-c:v': 'r210'
        # #     # '-c:v': 'copy',
        # # #     '-colorspace': '1'
        # # # #     '-f': 'rawvideo',
        # # # #     '-crf': '0',
        # #     # '-preset': 'veryslow'
        # })

        frames = np.float32(frames)

        print(type(frames), type(adv_frames))
        # frames = adv_frames
        # print(frames[[0, 1, 2, 3, 4, 5]].shape)
        plt.imshow(adv_frames[1]/255.)
        plt.show()

        plt.imshow(frames[1]/255.)
        plt.show()

        plt.imshow((adv_frames[1] - frames[1])/255.)
        plt.show()

        plt.imshow((adv_frames[1] - frames[1]))
        plt.show()

        print(adv_frames.shape, frames.shape)
        print(np.array(adv_frames).all() == np.array(frames).all())
        print(np.array_equal(np.array(adv_frames), np.array(adv_frames)))
        print(np.array_equal(np.array(adv_frames), np.array(frames)))
        print(np.mean(np.array(adv_frames)) - np.mean(np.array(frames)))
        print(np.mean(np.array(adv_frames)) - np.mean(np.array(adv_frames)))
        # print(np.array(adv_frames), np.array(frames))
        # print(np.array(adv_frames == frames).all() == True)
        # print(np.array(adv_frames == frames).any() == False)

        # bp ---
        # batches = create_batches(frames, load_img_fn, tf_img_fn)

        batches = exp_create_batches(frames, 3)
        feats = full_decoder.conv_forward((batches.unsqueeze(0)))
        seq_prob, seq_preds = full_decoder.encoder_decoder_forward(feats, mode='inference')

        # seq_prob, seq_preds = full_decoder(batches, mode='inference')
        sents = utils.decode_sequence(vocab, seq_preds)

        adv_caption = sents[0]


    print("\nOriginal Caption: {}\nTarget Caption: {}\nAdversarial Caption: {}".format(original_caption, target_caption, adv_caption))

# carlini = CarliniAttack(oracle=full_decoder, video_path=video_path, target=target_caption, dataset=dataset)

    # carlini.execute(video_path)

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
    print("new dims: {}".format(cropped_frames[0].shape))
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
    main(opt)