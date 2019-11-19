import os, sys, re
import numpy as np
import pandas as pd
import torch
import gzip
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn import preprocessing

colab_root = '/content/drive/My Drive/data_for_SNe_spec'
exalearn_root = '/home/jorgemarpa/data'
local_root = os.getcwd()

## load pkl synthetic light-curve files to numpy array
class DataSet_Class(Dataset):
    def __init__(self, machine='local', timestamp='111519',
                 length='1696'):
        """SNe Spec dataset"""

        if machine == 'local':
            root = '%s/data' % local_root
        elif machine == 'colab':
            root = colab_root
        elif machine == 'exalearn':
            root = '%s/spec/SNe' % exalearn_root
        else:
            print('Wrong machine, please select loca, colab or exalearn')
            sys.exit()

        self.spec_train = np.load('%s/spectraX.train.aug.%s.%s.class.npy'
                                  % (root, timestamp, length))
        self.label_train = np.load('%s/labels.train.aug.%s.%s.class.npy'
                                   % (root, timestamp, length),
                                   allow_pickle=True)


        self.spec_test = np.load('%s/spectraX.test.%s.%s.class.npy'
                                 % (root, timestamp, length))
        self.label_test = np.load('%s/labels.test.%s.%s.class.npy'
                                  % (root, timestamp, length),
                                  allow_pickle=True)

        self.spec_train = self.spec_train[:,:, np.newaxis].astype(np.float32)
        self.spec_test = self.spec_test[:,:, np.newaxis].astype(np.float32)

        self.label_int_enc = preprocessing.LabelEncoder()
        self.label_int_enc.fit(self.label_train)
        self.label_train_int = self.label_int_enc.transform(self.label_train)
        self.label_test_int = self.label_int_enc.transform(self.label_test)
        self.total_targets = len(set(self.label_train))

        self.spec_len = self.spec_test.shape[1]
        self.spec_nfeat = self.spec_test.shape[2]


    def __getitem__(self, index):
        spec = self.spec_train[index]
        label = self.label_train[index]
        label_int = self.label_train_int[index]
        return spec, label, label_int


    def __len__(self):
        return len(self.spec_train)


    def get_dataloader(self, batch_size=32, shuffle=True,
                       test_split=0.2, random_seed=42):

        if test_split == 0.:
            train_loader = DataLoader(self, batch_size=batch_size,
                                      shuffle=shuffle, drop_last=False)
            val_loader = None
        else:
            # Creating data indices for training and test splits:
            dataset_size = len(self)
            indices = list(range(dataset_size))
            split = int(np.floor(test_split * dataset_size))
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            train_indices, test_indices = indices[split:], indices[:split]

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)

            train_loader = DataLoader(self, batch_size=batch_size,
                                      sampler=train_sampler, drop_last=False)
            val_loader = DataLoader(self, batch_size=batch_size,
                                      sampler=test_sampler, drop_last=False)

        return train_loader, val_loader




class DataSet_Regr(Dataset):
    def __init__(self, machine='local', timestamp='111519',
                 length='1696'):
        """SNe Spec dataset"""

        if machine == 'local':
            root = '%s/data' % local_root
        elif machine == 'colab':
            root = colab_root
        elif machine == 'exalearn':
            root = '%s/spec/SNe' % exalearn_root
        else:
            print('Wrong machine, please select loca, colab or exalearn')
            sys.exit()

        self.spec_train = np.load('%s/spectraX.train.aug.%s.%s.regr.npy'
                                  % (root, timestamp, length))
        self.target_train = np.load('%s/labels.train.aug.%s.%s.regr.npy'
                                    % (root, timestamp, length),
                                    allow_pickle=True)[:,[0,2]]


        self.spec_test = np.load('%s/spectraX.test.%s.%s.regr.npy'
                                 % (root, timestamp, length))
        self.target_test = np.load('%s/labels.test.%s.%s.regr.npy'
                                   % (root, timestamp, length),
                                   allow_pickle=True)[:,[0,2]]

        self.spec_train = self.spec_train[:,:, np.newaxis].astype(np.float32)
        self.spec_test = self.spec_test[:,:, np.newaxis].astype(np.float32)
        self.target_train = self.target_train.astype(np.float32)
        self.target_test = self.target_test.astype(np.float32)

        self.scaler_data_min = np.array([-12, 0.6], dtype=np.float32)
        self.scaler_data_max = np.array([22, 1.8], dtype=np.float32)
        #self.target_train_n = (self.target_train - self.scaler_data_min)/\
        #                      (self.scaler_data_max - self.scaler_data_min)
        #self.target_test_n = (self.target_test - self.scaler_data_min)/\
        #                      (self.scaler_data_max - self.scaler_data_min)
        self.target_train[:,0] += 12
        self.target_test[:,0] += 12


        self.spec_len = self.spec_test.shape[1]
        self.spec_nfeat = self.spec_test.shape[2]

        self.names = ['phase', 'delta_m15']
        self.total_targets = 2


    def __getitem__(self, index):
        spec = self.spec_train[index]
        target = self.target_train[index]

        return spec, target


    def __len__(self):
        return len(self.spec_train)


    def get_dataloader(self, batch_size=32, shuffle=True,
                       test_split=0.2, random_seed=42):

        if test_split == 0.:
            train_loader = DataLoader(self, batch_size=batch_size,
                                      shuffle=shuffle, drop_last=False)
            val_loader = None
        else:
            # Creating data indices for training and test splits:
            dataset_size = len(self)
            indices = list(range(dataset_size))
            split = int(np.floor(test_split * dataset_size))
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            train_indices, test_indices = indices[split:], indices[:split]

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)

            train_loader = DataLoader(self, batch_size=batch_size,
                                      sampler=train_sampler, drop_last=False)
            val_loader = DataLoader(self, batch_size=batch_size,
                                      sampler=test_sampler, drop_last=False)

        return train_loader, val_loader
