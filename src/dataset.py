import os, sys
import numpy as np
import pandas as pd
import torch
import gzip
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

local_root = '/Users/jorgetil/Google Drive/Colab_Notebooks/data'
colab_root = '/content/drive/My Drive/Colab_Notebooks/data'
exalearn_root = '/home/jorgemarpa/data'

## load pkl synthetic light-curve files to numpy array
class DataSet(Dataset):
    def __init__(self, ):
        """EROS light curves data loader"""
        if machine == 'local':
            root = local_root
        elif machine == 'colab':
            root = colab_root
        elif machine == 'exalearn':
            root = exalearn_root
        else:
            print('Wrong machine, please select loca, colab or exalearn')
            sys.exit()

        # data_path = ('%s/time_series/real' % ())

        print('Loading from:\n', data_path)
        with gzip.open(data_path, 'rb') as f:
            self.aux = np.load(f, allow_pickle=True)
        # self.spec =
        # self.label =
        del self.aux


    def __getitem__(self, index):
        spec = self.spec[index]
        label = self.label[index]
        return spec, label


    def __len__(self):
        return len(self.lcs)


    def get_dataloader(self, batch_size=32, shuffle=True,
                       test_split=0.2, random_seed=42):

        if test_split == 0.:
            train_loader = DataLoader(self, batch_size=batch_size,
                                      shuffle=shuffle, drop_last=False)
            test_loader = None
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
            test_loader = DataLoader(self, batch_size=batch_size,
                                      sampler=test_sampler, drop_last=False)

        return train_loader, test_loader
