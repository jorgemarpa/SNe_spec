import torch
import torch.nn as nn
import torch.nn.functional as F


class RecNN(nn.Module):
    def __init__(self, seq_len, n_feats, hidden_dim, n_layers,
                 rnn='lstm', out_size=1,
                 dropout=0.3, bidir=False):
        super(RecNN, self).__init__()

        ## batch_first --> [batch, seq, feature]
        if rnn == 'lstm':
            self.rnn = nn.LSTM(n_feats, hidden_dim, n_layers,
                              dropout=dropout, bidirectional=bidir,
                              batch_first=True)
        elif rnn == 'gru':
            self.rnn = nn.GRU(n_feats, hidden_dim, n_layers,
                              dropout=dropout, bidirectional=bidir,
                              batch_first=True)
        else:
            self.rnn = nn.RNN(n_feats, hidden_dim, n_layers,
                              nonlinearity='relu',
                              dropout=dropout, bidirectional=bidir,
                              batch_first=True)

        self.linear = nn.Linear(hidden_dim * (2 if bidir else 1),
                                out_size)

        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.init_weights()


    def init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param, mean=0.0, std=1.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)


    def forward(self, x):
        out, h = self.rnn(x)
        out = self.log_softmax(self.linear(out[:, -1, :]))
        return out


# class ConvNN(nn.Module):
#     def __init__(self, seq_len, hidden_dim, n_layers,
#                  n_feats=3, dropout=0.3):
#         super(ConvNN, self).__init__()
#
#         self.conv1 = nn.Conv1d()
#         self.conv1 = nn.Conv1d()
#         self.conv1 = nn.Conv1d()
#
#         self.bn1 = nn.BatchNorm1d()
#         self.bn2 = nn.BatchNorm1d()
#         self.bn3 = nn.BatchNorm1d()
#
#         self.dp1 = nn.Dropout(dropout)
#         self.dp1 = nn.Dropout(dropout)
#         self.dp1 = nn.Dropout(dropout)
#
#         self.maxpool = nn.MaxPool1d()
#         self.maxpool = nn.MaxPool1d()
#         self.maxpool = nn.MaxPool1d()
#
#         self.out = nn.Linear()
#
#         self.init_weights()
#
#
#     def forward(self, x):
#         # out =
#         return out
