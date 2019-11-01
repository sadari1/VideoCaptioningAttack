import torch.nn as nn
import torch
import numpy as np
import pretrainedmodels
from ..process_features import process_batches, create_batches


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
        self.s2vt.load_state_dict(torch.load(opt["saved_model"]))

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
