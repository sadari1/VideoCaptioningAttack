import json
import os
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from dataloader import VideoDataset
import misc.utils as utils
from misc.cocoeval import suppress_stdout_stderr, COCOScorer
from collections import OrderedDict, defaultdict
from pandas.io.json import json_normalize


def convert_data_to_coco_scorer_format(vid_ids, vid_to_meta):
    gts = defaultdict(list)

    for i in vid_ids:
        captions = vid_to_meta[i]["final_captions"]
        for j, cap in enumerate(captions):
            gts[i].append({'image_id': i, 'cap_id': j, 'caption': ' '.join(cap)})

    return gts


def test(model, crit, dataset, vocab, opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    loader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=False)
    scorer = COCOScorer()
    dataset_meta = json.load(open(opt["dataset_json"]))

    vid_to_meta = dataset_meta["vid_to_meta"]
    vid_ids = dataset_meta["split_to_ids"]["test"]

    # gt_dataframe = json_normalize(json.load(open(opt["dataset_json"]))['sentences'])
    gts = convert_data_to_coco_scorer_format(vid_ids, vid_to_meta)
    #results = []
    samples = {}

    with torch.no_grad():
        for index, data in enumerate(loader):
            print('batch: '+str((index+1)*opt["batch_size"]))
            # forward the model to get loss
            fc_feats = data['fc_feats'].to(device)
            video_id = data['video_ids'].cpu()

            # forward the model to also get generated samples for each image
            seq_probs, seq_preds = model(fc_feats, mode='inference', opt=opt)

            sents = utils.decode_sequence(vocab, seq_preds)

            for k, sent in enumerate(sents):
                # Iter through each video in batch and convert id back to original msvd key
                vid_key = vid_ids[video_id[k]]
                samples[vid_key] = [{'image_id': vid_key, 'caption': sent}]

    with suppress_stdout_stderr():
        valid_score = scorer.score(gts, samples, samples.keys())

    print(valid_score)

    # TODO: Make this usable eventually
    # if not os.path.exists(opt["results_path"]):
    #     os.makedirs(opt["results_path"])
    #
    # result = OrderedDict()
    # result['checkpoint'] = opt["saved_model"][opt["saved_model"].rfind('/')+1:]
    #
    # score_sum = 0
    #
    # for key, value in valid_score.items():
    #     score_sum += float(value)
    #
    # result['sum'] = str(score_sum)
    #
    # #result = OrderedDict(result, **valid_score)
    # result = OrderedDict(result.items() + valid_score.items())
    # print(result)
    # if not os.path.exists(opt["results_path"]):
    #     os.makedirs(opt["results_path"])
    # with open(os.path.join(opt["results_path"], "scores.txt"), 'a') as scores_table:
    #     scores_table.write(json.dumps(result) + "\n")
    # with open(os.path.join(opt["results_path"],
    #                        opt["model"].split("/")[-1].split('.')[0] + ".json"), 'w') as prediction_results:
    #     json.dump({"predictions": samples, "scores": valid_score},
    #               prediction_results)


def main(opt):
    dataset = VideoDataset(opt, "test")
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

    if torch.cuda.device_count() > 1:
        print("{} devices detected, switch to parallel model.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Setup the model
    model.load_state_dict(torch.load(opt["saved_model"]))
    crit = utils.LanguageModelCriterion()

    test(model, crit, dataset, dataset.get_vocab(), opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
