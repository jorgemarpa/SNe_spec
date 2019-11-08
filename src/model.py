import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, seq_len, hidden_dim, n_layers,
                 rnn='lstm', n_feats=3, dropout=0.3):
        super(RNN, self).__init__()

        self.seq_len = seq_len

        self.input = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * seq_len),
            nn.ReLU())

        ## batch_first --> [batch, seq, feature]
        if rnn == 'lstm':
            self.rnn = nn.LSTM(n_feats, hidden_dim, n_layers,
                                    batch_first=True, dropout=dropout,
                                    bidirectional=False)
        elif rnn == 'gru':
            self.rnn = nn.GRU(n_feats, hidden_dim, n_layers,
                                   batch_first=True, dropout=dropout,
                                   bidirectional=False)

        self.enc_linear_mu = nn.Linear(seq_len * hidden_dim + con_dim,
                                       latent_dim)

        self.init_weights()


    def init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)


    def forward(self, x, c=None):
        # out =
        return out
