from utils import *

import json
import os
import argparse
import torch
import numpy as np
import PIL
import skvideo

from video_caption_pytorch.models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from video_caption_pytorch.dataloader import VideoDataset
from video_caption_pytorch.models.ConvS2VT import ConvS2VT
from global_constants import *
import matplotlib.pyplot as plt
from video_attack import CarliniAttack, create_batches
from pretrainedmodels import utils as ptm_utils
from video_caption_pytorch.misc import utils as vcp_utils

np.random.seed(SEED)
torch.manual_seed(SEED)

'''
python experiment_video_attack_rand_test_caption.py
"D:\College\Research\videostoattack" 
"D:\College\Research\attacked" 
--recover_opt "D:\College\Research\December 2018 Video Captioning Attack\video captioner\save\msvd_nasnetalarge\opt_info.json" 
--saved_model "D:\College\Research\December 2018 Video Captioning Attack\video captioner\save\msvd_nasnetalarge\model_1000.pth"
'''


def main(opt):
    dataset = VideoDataset(opt, 'inference')
    opt["vocab_size"] = dataset.get_vocab_size()
    opt["seq_length"] = dataset.max_len
    vocab = dataset.get_vocab()

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

    convnet = 'nasnetalarge'
    full_decoder = ConvS2VT(convnet, model, opt)

    video_path = opt['videos'][0]
    vid_id = video_path.split('/')[-1]
    vid_id = vid_id.split('.')[0]
    # orig_captions = [' '.join(toks) for toks in dataset.vid_to_meta[vid_id]['final_captions']]

    viable_ids = dataset.splits['test'] + dataset.splits['val']
    viable_target_captions = []
    for v_id in viable_ids:
        if v_id == vid_id:
            continue
        plausible_caps = [' '.join(toks) for toks in dataset.vid_to_meta[v_id]['final_captions']]
        viable_target_captions.extend(plausible_caps)

    target_caption = np.random.choice(viable_target_captions)
    interval = BATCH_SIZE

    num_seconds = 0.5
    numIt = 4 # int(24 * num_seconds)
    real_len = len(skvideo.io.vread(video_path))
    assert numIt <= real_len

    print("\t\t{} iterations to do.".format(numIt))
    counter = 0
    totalframes = []
    adv_batches = []

    while numIt > (interval - 1):
        window = range(counter, counter+interval)
        counter += interval
        carlini = CarliniAttack(oracle=full_decoder, video_path=video_path, target=target_caption, dataset=dataset, window=window)
        frames = carlini.execute(video_path, window=window, functional=True)
        totalframes.append(frames.detach().cpu().numpy())
        adv_batches.append(create_batches(frames, batch_size=interval).detach().cpu().numpy())
        numIt -= interval
        print("\t\tWindow {}".format(numIt))

    if numIt > 0:
        window = range(counter, counter + numIt)
        carlini = CarliniAttack(oracle=full_decoder, video_path=video_path, target=target_caption, dataset=dataset, window=window)
        frames = carlini.execute(video_path, window=window, functional=True)
        totalframes.append(frames.detach().cpu().numpy())
        adv_batches.append(create_batches(frames, batch_size=interval).detach().cpu().numpy())
        print("\t\tWindow {}".format(numIt))

    base_toks = video_path.split('/')
    base_dir_toks = base_toks[:-1]
    base_filename = base_toks[-1]
    base_name = ''.join(base_filename.split('.')[:-1])
    adv_path = os.path.join('/'.join(base_dir_toks), base_name + '_adversarial.avi')

    frames = np.concatenate(totalframes, axis=0)
    save_frames_to_video(frames, adv_path)

    batches = np.concatenate(adv_batches, axis=0)

    with torch.no_grad():
        print(frames.shape)

        # bp ---
        seq_prob, seq_preds = full_decoder(batches, mode='inference', single_batch=False)
        sents = vcp_utils.decode_sequence(vocab, seq_preds)

        print(sents[0])


def save_frames_to_video(batched_t, fpath):
    # in_frames = batched_t.detach().cpu().numpy()
    print("Saving video to: %s" % fpath)
    skvideo.io.vwrite(fpath, batched_t)


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