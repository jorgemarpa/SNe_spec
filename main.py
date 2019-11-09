import sys
import datetime
import argparse
import torch
import torch.optim as optim
import numpy as np
from src.dataset import DataSet
from src.trainer import Trainer
from src.model import *

import wandb

rnd_seed = 13
torch.manual_seed(rnd_seed)
np.random.seed(rnd_seed)

## Config ##
parser = argparse.ArgumentParser(description='Conv/RNN model to classify'+
                                 'SNe Spectra into a Phase/delta-m_15 grid')
parser.add_argument('--dry-run', dest='dry_run', action='store_true',
                    default=False,
                    help='Load data and initialize models [False]')
parser.add_argument('--run-name', dest='run_name', type=str, default='test',
                    help='Name of the run')
parser.add_argument('--machine', dest='machine', type=str, default='local',
                    help='Where is running ([local], colab, exalearn)')

parser.add_argument('--data', dest='data', type=str, default='spec',
                    help='Which data use ([spec])')

parser.add_argument('--optim', dest='optim', type=str, default='adam',
                    help='Optimizer ([adam],sdg)')
parser.add_argument('--lr', dest='lr', type=float, default=1e-3,
                    help='learning rate [1e-3]')
parser.add_argument('--lr-sch', dest='lr_sch', type=str, default=None,
                    help='learning rate shceduler '+
                    '([None], step, exp, cosine, plateau)')
parser.add_argument('--early-stop', dest='early_stop', action='store_true',
                    default=False, help='Early stopping, [False]')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=64,
                    help='batch size [64]')
parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=200,
                    help='total number of training epochs [200]')

parser.add_argument('--arch', dest='arch', type=str, default='lstm',
                    help='architecture for Enc & Dec ([lstm],gru,rnn,conv)')
parser.add_argument('--hidden-units', dest='h_units', type=int, default=32,
                    help='number of hidden units [32]')
parser.add_argument('--rnn-layers', dest='rnn_layers', type=int, default=3,
                    help='number of layers for rnn [5]')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.3,
                    help='dropout for rnn/conv layers [0.2]')
parser.add_argument('--rnn-bidir', dest='rnn_bidir', action='store_true',
                    default=False, help='Bidirectional RNN [False]')
parser.add_argument('--kernel-size', dest='kernel_size', type=int, default=5,
                    help='kernel size for conv, use odd ints [5]')

parser.add_argument('--comment', dest='comment', type=str, default='',
                    help='extra comments')
args = parser.parse_args()

# Initialize W&B project
wandb.init(project="SNe_Spec", name=args.run_name)

# save hyper-parameters to W&B
wandb.config.update(args)
wandb.config.rnd_seed = rnd_seed


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


## run main program
def run_code():

    ## Load Data ##
    dataset = DataSet()

    wandb.config.spec_len = dataset.spec_len
    wandb.config.spec_nfeat = dataset.spec_nfeat
    wandb.config.total_classes = dataset.total_cls

    if len(dataset) == 0:
        print('No items in training set...')
        print('Exiting!')
        sys.exit()

    ## data loaders for training and testing
    train_loader, test_loader = dataset.get_dataloader(batch_size=args.batch_size,
                                                       shuffle=rnd_seed,
                                                       test_split=.2,
                                                       random_seed=rnd_seed)
    print('')
    print('Training lenght  : ', len(train_loader) * args.batch_size)
    print('Validation lenght: ', len(test_loader) * args.batch_size)

    ## Define GAN model, Ops, and Train ##
    if args.arch in ['lstm','gru','rnn']:
        model = RecNN(seq_len=dataset.spec_len,
                      n_feats=dataset.spec_nfeat,
                      hidden_dim=args.h_units,
                      n_layers=args.rnn_layers,
                      rnn=args.arch,
                      out_size=dataset.total_cls,
                      dropout=args.dropout,
                      bidir=args.rnn_bidir)
    elif args.arch == 'conv':
        print('Not implemented yet...')
        print('Exiting...')
        sys.exit()

    wandb.watch(model, log='gradients')

    print('Summary:')
    print(model)

    # print(generator)
    wandb.config.n_train_params = count_parameters(model)
    print('\n')
    print('Num of trainable params: ', count_parameters(model))
    print('\n')

    # Initialize optimizers
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'sdg':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=.9)

    # Learning Rate scheduler
    if args.lr_sch == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=20,
                                              gamma=0.5)
    elif args.lr_sch == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=0.985)
    elif args.lr_sch == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=50,
                                                         eta_min=1e-5)
    elif args.lr_sch == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=.5,
                                                         verbose=True)
    else:
        scheduler = None

    print('Optimizer    :', optimizer)
    print('LR Scheduler :', scheduler.__class__.__name__)
    print('\n')

    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print('########################################')
    print('########  Running in %4s  #############' % (device))
    print('########################################')

    trainer = Trainer(model, optimizer, args.batch_size, wandb,
                      print_every=50,
                      device=device,
                      scheduler=scheduler)

    if args.dry_run:
        print('******** DRY RUN ******** ')
        return

    trainer.train(train_loader, test_loader, args.num_epochs,
                  save=True, early_stop=args.early_stop)


if __name__ == "__main__":
    for key, value in vars(args).items():
        print('%15s\t: %s' % (key, value))

    run_code()
