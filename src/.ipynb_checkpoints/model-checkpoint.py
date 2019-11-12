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

        self.log_softmax = nn.LogSoftmax(dim=1)

        self.init_weights()


    def init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param, mean=0.0, std=1.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)


    def forward(self, x):
        self.rnn.flatten_parameters()
        out, h = self.rnn(x)
        out = self.log_softmax(self.linear(out[:, -1, :]))
        return out



class ConvNN(nn.Module):
    def __init__(self, seq_len, n_feats,
                 kernel_size=3, out_size=1, dropout=0.3):
        super(ConvNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_feats, 4, kernel_size),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
            )
        self.conv2 = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
            )
        self.conv3 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
            )
        self.conv4 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
            )

        self.out = nn.Linear(32 * 19, out_size)
        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        # shape(x) = N, L, C
        # conv1d needs N,C,L
        #print(x.shape)
        x = x.transpose(1,2)
        #print('transp', x.shape)
        x = self.conv1(x)
        #print('cnv1', x.shape)
        x = self.conv2(x)
        #print('cnv2', x.shape)
        x = self.conv3(x)
        #print('cnv3', x.shape)
        x = self.conv4(x)
        #print('cnv4', x.shape)
        
        x = x.flatten(start_dim=1)
        #print('flat', x.shape)
        
        out = self.out(x)
        #print('out', out.shape)
        log_prob = self.log_softmax(out)
        return log_prob
