from utils import *
import skvideo
import json
import os
import torch
import argparse
import numpy as np

from video_caption_pytorch.models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from video_caption_pytorch.dataloader import VideoDataset
from video_caption_pytorch.models.ConvS2VT import ConvS2VT
from global_constants import *
from utils import *
from video_attack import CarliniAttack

np.random.seed(SEED)
torch.manual_seed(SEED)

'''
"D:\College\Research\videostoattack" 
"D:\College\Research\attacked" 
--recover_opt "D:\College\Research\December 2018 Video Captioning Attack\video captioner\save\msvd_nasnetalarge\opt_info.json" 
--saved_model "D:\College\Research\December 2018 Video Captioning Attack\video captioner\save\msvd_nasnetalarge\model_1000.pth"

'''


def main(opt):
    dataset = VideoDataset(opt, 'inference')
    time_stamp = get_time_stamp()

    if not os.path.isdir(os.path.join(opt['adv_dir'], time_stamp)):
        os.makedirs(os.path.join(opt['adv_dir'], time_stamp))

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

    convnet = 'nasnetalarge'
    full_decoder = ConvS2VT(convnet, model, opt)

    video_names = os.listdir(opt['from_dir'])

    viable_ids = dataset.splits['test'] + dataset.splits['val']

    for vn in video_names:
        video_path = os.path.join(opt['from_dir'], vn)
        vid_id = video_path.split('\\')[-1]
        vid_id = vid_id.split('.')[0]
        orig_captions = [' '.join(toks) for toks in dataset.vid_to_meta[vid_id]['final_captions']]
        original_caption = np.random.choice(orig_captions)

        viable_target_captions = []
        for v_id in viable_ids:
            if v_id == vid_id:
                continue
            plausible_caps = [' '.join(toks) for toks in dataset.vid_to_meta[v_id]['final_captions'] if len(toks) <= MAX_TARGET_LEN]
            viable_target_captions.extend(plausible_caps)

        target_caption = np.random.choice(viable_target_captions)

        carlini = CarliniAttack(oracle=full_decoder, video_path=video_path, target=target_caption, dataset=dataset)

        stats_obj = carlini.execute(video_path, functional=True, stats=True)
        stats_obj['original_caption'] = original_caption
        stats_obj['target_caption'] = target_caption

        base_name = ''.join(vn.split('.')[:-1])
        adv_path = os.path.join(opt['adv_dir'], time_stamp, base_name + '_adversarial.avi')
        adv_raw_path = os.path.join(opt['adv_dir'], time_stamp, base_name + '_adversarial.pkl')

        save_tensor_to_video(stats_obj['pass_in'], adv_path)
        save_tensor_to_video(stats_obj['delta'], adv_path)
        pickle_write(adv_raw_path, stats_obj)


def save_tensor_to_video(batched_t, fpath):
    in_frames = batched_t
    skvideo.io.vwrite(fpath, in_frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('from_dir', type=str,
                        help='')
    parser.add_argument('adv_dir', type=str,
                        help='')
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