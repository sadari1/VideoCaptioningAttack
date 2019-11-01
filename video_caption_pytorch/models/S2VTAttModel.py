import torch.nn as nn


class S2VTAttModel(nn.Module):
    def __init__(self, encoder, decoder):
        """

        Args:
            encoder (nn.Module): Encoder rnn
            decoder (nn.Module): Decoder rnn
        """
        # python 3
        # super().__init__()
        super(S2VTAttModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, vid_feats, target_variable=None,
                mode='train', get_attn=False, opt={}):
        """

        Args:
            vid_feats (Variable): video feats of shape [batch_size, seq_len, dim_vid]
            target_variable (None, optional): groung truth labels

        Returns:
            seq_prob: Variable of shape [batch_size, max_len-1, vocab_size]
            seq_preds: [] or Variable of shape [batch_size, max_len-1]
        """


        encoder_outputs, encoder_hidden = self.encoder(vid_feats)
        if (get_attn):
            attn = self.decoder(encoder_outputs=encoder_outputs, get_attn=get_attn, encoder_hidden=encoder_hidden, targets=target_variable, mode=mode, opt=opt)
            return attn

        seq_prob, seq_preds = self.decoder(encoder_outputs, encoder_hidden, target_variable, mode, opt)
        return seq_prob, seq_preds
