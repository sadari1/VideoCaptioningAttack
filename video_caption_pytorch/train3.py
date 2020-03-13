import json
import os

import numpy as np

import argparse
from misc import utils as utils
from opts import parse_opt
import torch
import torch.optim as optim
from dataloader import VideoDataset
from misc.rewards import get_self_critical_reward, init_cider_scorer
from models import DecoderRNN, EncoderRNN, S2VTAttModel, S2VTModel
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

'''


python train3.py --recover_opt "D:/College/Research/December 2018 Video Captioning Attack/video captioner/save/msvd_nasnetalarge/opt_info.json"
python train3.py --recover_opt "D:/College/Research/December 2018 Video Captioning Attack/video captioner/save/msvd_resnet152/opt_info.json"
python train3.py --recover_opt "D:/College/Research/December 2018 Video Captioning Attack/video captioner/save/msvd_vgg16/opt_info.json"
'''

'''Trained for 1112 iterations, the loss is at: 8.82 to 10.22 to 11.39'''
def train(loader, model, crit, optimizer, lr_scheduler, opt, rl_crit=None):
    model.train()
    if torch.cuda.device_count() > 1:
        print("{} devices detected, switch to parallel model.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(opt["epochs"]):
        lr_scheduler.step()

        iteration = 0
        # If start self crit training
        if opt["self_crit_after"] != -1 and epoch >= opt["self_crit_after"]:
            sc_flag = True
            init_cider_scorer(opt["cached_tokens"])
        else:
            sc_flag = False

        for data in loader:
            torch.cuda.synchronize()
            fc_feats = data['fc_feats'].to(device)
            labels = data['labels'].to(device)
            masks = data['masks'].to(device)

            if not sc_flag:
                seq_probs, _ = model(fc_feats, labels, 'train')
                loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
            else:
                seq_probs, seq_preds = model(fc_feats, mode='inference', opt=opt)
                reward = get_self_critical_reward(model, fc_feats, data, seq_preds)
                print(reward.shape)
                loss = rl_crit(seq_probs, seq_preds,
                               Variable(
                                   torch.from_numpy(reward).float().cuda()))

            optimizer.zero_grad()
            loss.backward()
            utils.clip_gradient(optimizer, opt["grad_clip"])
            optimizer.step()
            train_loss = loss.data[0]
            torch.cuda.synchronize()
            iteration += 1

            if not sc_flag:
                print("iter %d (epoch %d), train_loss = %.6f" %
                      (iteration, epoch, train_loss))
            else:
                print("iter %d (epoch %d), avg_reward = %.6f" %
                      (iteration, epoch, np.mean(reward[:, 0])))

        if epoch != 0 and epoch % opt["save_checkpoint_every"] == 0:
            model_path = os.path.join(opt["checkpoint_path"],
                                      'vgg16_model_%d.pth' % (epoch))
            model_info_path = os.path.join(opt["checkpoint_path"],
                                           'vgg16_model_score.txt')
            torch.save(model.state_dict(), model_path)
            print("model saved to %s" % (model_path))
            with open(model_info_path, 'a') as f:
                f.write("resnet_model_%d, loss: %.6f\n" % (epoch, train_loss))


def main(opt):



    dataset = VideoDataset(opt, 'train')
    dataloader = DataLoader(dataset,
                            batch_size=opt["batch_size"],
                            num_workers=8,
                            shuffle=True)
    opt["vocab_size"] = dataset.get_vocab_size()
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            opt['dim_vid'],
            rnn_cell=opt['rnn_type'],
            n_layers=opt['num_layers'],
            bidirectional=opt["bidirectional"],
            rnn_dropout_p=opt["rnn_dropout_p"]).cuda()
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(
            opt["dim_vid"],
            opt["dim_hidden"],
            n_layers=opt['num_layers'],
            bidirectional=opt["bidirectional"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"])
        decoder = DecoderRNN(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            n_layers=opt['num_layers'],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"],
            bidirectional=opt["bidirectional"])
        model = S2VTAttModel(encoder, decoder).cuda()
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

    model.load_state_dict(torch.load("C:\\Users\\Shumpu\\VideoCaptioningAttack\\video_caption_pytorch\\save\\vgg16_model_460.pth"))
    train(dataloader, model, crit, optimizer, exp_lr_scheduler, opt, rl_crit)


if __name__ == '__main__':
    # opt = parse_opt()
    # opt = vars(opt)
    # for key, value in opt.items():
    #     print(key, value)

    parser = argparse.ArgumentParser()
    parser.add_argument('--recover_opt', type=str, required=True,
                        help='recover train opts from saved opt_json')

    args = parser.parse_args()
    args = vars((args))

    opt = json.load(open(args["recover_opt"]))
    # args = parser.parse_args()
    # args = vars((args))
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    # opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
    # if not os.path.exists(opt["checkpoint_path"]):
    #     os.makedirs(opt["checkpoint_path"])
    # with open(opt_json, 'w') as f:
    #     json.dump(opt, f)
    # print('save opt details to %s' % (opt_json))
    try:
        main(opt)
    except Exception as e:
        raise e
