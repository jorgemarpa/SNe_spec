import os, sys, re
import numpy as np
import pandas as pd
import torch
import gzip
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn import preprocessing

colab_root = '/content/drive/My Drive/Colab_Notebooks/data'
exalearn_root = '/home/jorgemarpa/data'
local_root = os.getcwd()

## load pkl synthetic light-curve files to numpy array
class DataSet(Dataset):
    def __init__(self, machine='local'):
        """SNe Spec dataset"""

        if machine == 'local':
            root = local_root
        elif machine == 'colab':
            root = colab_root
        elif machine == 'exalearn':
            root = exalearn_root
        else:
            print('Wrong machine, please select loca, colab or exalearn')
            sys.exit()

        self.spec_train = np.load('%s/data/iter_1/spectraX.train.aug.110819.npy' % (root))
        self.label_train = np.load('%s/data/iter_1/labels.train.aug.110819.npy' % (root))


        self.spec_test = np.load('%s/data/iter_1/spectraX.test.110819.npy' % (root))
        self.label_test = np.load('%s/data/iter_1/labels.test.110819.npy' % (root))

        self.spec_train = self.spec_train[:,:, np.newaxis].astype(np.float32)
        self.spec_test = self.spec_test[:,:, np.newaxis].astype(np.float32)

        self.label_int_enc = preprocessing.LabelEncoder()
        self.label_int_enc.fit(self.label_train)
        self.label_train_int = self.label_int_enc.transform(self.label_train)
        self.label_test_int = self.label_int_enc.transform(self.label_test)

        self.spec_len = self.spec_test.shape[1]
        self.spec_nfeat = self.spec_test.shape[2]
        self.total_cls = len(set(self.label_train))


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
