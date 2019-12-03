import torch
import torch.nn as nn
import torch.nn.functional as F


class RecNN_Clas(nn.Module):
    def __init__(self, seq_len, n_feats, hidden_dim, n_layers,
                 rnn='lstm', out_size=1, dropout=0.3,
                 bidir=False, device='cpu'):
        super(RecNN_Clas, self).__init__()

        self.rnn_s = rnn
        self.n_layers = n_layers
        self.bidir = bidir
        self.hidden_dim = hidden_dim
        self.device = device

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

        #self.log_softmax = nn.LogSoftmax(dim=1)

        self.init_weights()


    def init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                if self.rnn_s == 'lstm':
                    nn.init.constant_(param, 1)
                elif self.rnn_s == 'gru':
                    nn.init.constant_(param, -1)
                else:
                    nn.init.uniform_(param, -1, 1)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)


    def forward(self, x):
        self.rnn.flatten_parameters()
        h_0 = torch.randn((self.n_layers * (2 if self.bidir else 1),
                           x.shape[0],
                           self.hidden_dim)).to(self.device)
        if self.rnn_s == 'lstm':
            c_0 = torch.randn((self.n_layers * (2 if self.bidir else 1),
                               x.shape[0],
                               self.hidden_dim)).to(self.device)
            x, h = self.rnn(x, (h_0, c_0))
        else:
            x, h = self.rnn(x, h_0)
        #x = x.flatten(start_dim=1)
        x = x[:,-1,:]
        out = self.linear(x)

        log_prob = F.log_softmax(out, dim=1)
        return log_prob



class RecNN_Regr(nn.Module):
    def __init__(self, seq_len, n_feats, hidden_dim, n_layers,
                 rnn='lstm', out_size=1, dropout=0.3,
                 bidir=False, device='cpu'):
        super(RecNN_Regr, self).__init__()

        self.rnn_s = rnn
        self.n_layers = n_layers
        self.bidir = bidir
        self.hidden_dim = hidden_dim
        self.device = device

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

        ## if doing many-to-many, input size must be x seq_len
        self.linear_p = nn.Linear(hidden_dim * (2 if bidir else 1),
                                  1)
        self.linear_dm = nn.Linear(hidden_dim * (2 if bidir else 1),
                                  1)

        self.init_weights()


    def init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                if self.rnn_s == 'lstm':
                    nn.init.constant_(param, 1)
                elif self.rnn_s == 'gru':
                    nn.init.constant_(param, -1)
                else:
                    nn.init.uniform_(param, -1, 1)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)


    def forward(self, x):
        self.rnn.flatten_parameters()
        h_0 = torch.randn((self.n_layers * (2 if self.bidir else 1),
                           x.shape[0],
                           self.hidden_dim)).to(self.device)
        if self.rnn_s == 'lstm':
            c_0 = torch.randn((self.n_layers * (2 if self.bidir else 1),
                               x.shape[0],
                               self.hidden_dim)).to(self.device)
            x, h = self.rnn(x, (h_0, c_0))
        else:
            x, h = self.rnn(x, h_0)
        ## if using many-to-many, then x need to be flattend in the last 2 dim
        #x = x.flatten(start_dim=1)
        ## if doing many-to-one, then only use last L in x.shape=[N,L,C]
        x = x[:,-1,:]
        phase = self.linear_p(x)
        dm = self.linear_dm(x)

        return phase, dm



class ConvNN_Clas(nn.Module):
    def __init__(self, seq_len, n_feats,
                 n_blocks=3, hidden_channels=4,
                 kernel_size=3, out_size=1, dropout=0.3):
        super(ConvNN_Clas, self).__init__()

        if type(hidden_channels) == int:
            self.hidden_channels = [hidden_channels] * n_blocks
        else:
            self.hidden_channels = hidden_channels

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_feats,
                      self.hidden_channels[0],
                      kernel_size),
            nn.BatchNorm1d(self.hidden_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size)
            )
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.hidden_channels[0],
                      self.hidden_channels[1],
                      kernel_size),
            nn.BatchNorm1d(self.hidden_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size)
            )
        self.conv3 = nn.Sequential(
            nn.Conv1d(self.hidden_channels[1],
                      self.hidden_channels[2],
                      kernel_size),
            nn.BatchNorm1d(self.hidden_channels[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size)
            )
        #self.conv4 = nn.Sequential(
        #    nn.Conv1d(self.hidden_channels[2],
        #              self.hidden_channels[3],
        #              kernel_size),
        #    nn.BatchNorm1d(self.hidden_channels[3]),
        #    nn.ReLU(),
        #    nn.MaxPool1d(kernel_size=kernel_size)
        #    )

        def maxpool_out(l0, k, st):
            return int((l0 - k)/st)

        self.cnv_l = seq_len
        for i in range(len(self.hidden_channels)):
            self.cnv_l = maxpool_out(self.cnv_l,
                                     kernel_size,
                                     kernel_size)
        if kernel_size == 2:
            self.cnv_l += 1

        self.out = nn.Linear(self.hidden_channels[-1] * self.cnv_l,
                             out_size)



    def forward(self, x):
        # shape(x) = N, L, C
        # conv1d needs N,C,L
        x = x.transpose(1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #x = self.conv4(x)

        x = x.flatten(start_dim=1)
        out = self.out(x)

        return F.log_softmax(out, dim=1)



class ConvNN_Regr(nn.Module):
    def __init__(self, seq_len, n_feats,
                 n_blocks=5, hidden_channels=4,
                 kernel_size=3, out_size=1, dropout=0.3):
        super(ConvNN_Regr, self).__init__()

        if type(hidden_channels) == int:
            self.hidden_channels = [hidden_channels] * n_blocks
        else:
            self.hidden_channels = hidden_channels

        pool_size = 3

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_feats,
                      self.hidden_channels[0],
                      kernel_size),
            nn.BatchNorm1d(self.hidden_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size)
            )
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.hidden_channels[0],
                      self.hidden_channels[1],
                      kernel_size),
            nn.BatchNorm1d(self.hidden_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size)
            )
        self.conv3 = nn.Sequential(
            nn.Conv1d(self.hidden_channels[1],
                      self.hidden_channels[2],
                      kernel_size),
            nn.BatchNorm1d(self.hidden_channels[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size)
            )
        self.conv4 = nn.Sequential(
            nn.Conv1d(self.hidden_channels[2],
                      self.hidden_channels[3],
                      kernel_size),
            nn.BatchNorm1d(self.hidden_channels[3]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size)
            )
        self.conv5 = nn.Sequential(
            nn.Conv1d(self.hidden_channels[3],
                      self.hidden_channels[4],
                      kernel_size),
            nn.BatchNorm1d(self.hidden_channels[4]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size)
            )

        def maxpool_out(l0, k, st):
            return int((l0 - k)/st + 1)

        self.cnv_l = seq_len
        for i in range(len(self.hidden_channels)):
            self.cnv_l = maxpool_out(self.cnv_l,
                                     kernel_size,
                                     1)
            print('cnv  %i: ' % i, self.cnv_l)
            self.cnv_l = maxpool_out(self.cnv_l,
                                     pool_size,
                                     pool_size)
            print('pool %i: ' % i, self.cnv_l)

        self.fc_hp = nn.Linear(self.hidden_channels[-1] * self.cnv_l, 16)
        self.fc_hdm = nn.Linear(self.hidden_channels[-1] * self.cnv_l, 16)

        self.out_p = nn.Linear(16, 1)
        self.out_dm = nn.Linear(16, 1)


    def forward(self, x):
        # shape(x) = N, L, C
        # conv1d needs N,C,L
        x = x.transpose(1,2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        print(x.shape, self.cnv_l)

        x = x.flatten(start_dim=1)
        print(x.shape)
        h_phase = F.celu(self.fc_hp(x))
        h_dm = F.celu(self.fc_hdm(x))
        ## add activation layers if desired:
        # F.relu(self.out_p(h)) or F.sigmoid(self.out_p(h))
        phase = self.out_p(h_phase)
        dm = self.out_dm(h_dm)

        return phase, dm
