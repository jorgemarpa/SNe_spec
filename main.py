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
parser = argparse.ArgumentParser(description='Variational Auto Encoder (VAE)'+
                                 'to produce synthetic astronomical time series')
parser.add_argument('--dry-run', dest='dry_run', action='store_true',
                    default=False,
                    help='Load data and initialize models [False]')
parser.add_argument('--run-name', dest='run_name', type=str, default='test',
                    help='name of the run')
parser.add_argument('--machine', dest='machine', type=str, default='local',
                    help='were to is running ([local], colab, exalearn)')

parser.add_argument('--data', dest='data', type=str, default='eros',
                    help='data used for training ([eros])')

parser.add_argument('--lr', dest='lr', type=float, default=1e-4,
                    help='learning rate [1e-4]')
parser.add_argument('--lr-sch', dest='lr_sch', type=str, default=None,
                    help='learning rate shceduler '+
                    '([None], step, exp,cosine, plateau)')
parser.add_argument('--early-stop', dest='early_stop', action='store_true',
                    default=False,
                    help='Early stopping, default:False')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=64,
                    help='batch size [64]')
parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=200,
                    help='total number of training epochs [200]')

parser.add_argument('--arch', dest='arch', type=str, default='lstm',
                    help='architecture for Enc & Dec ([lstm],gru)')
parser.add_argument('--hidden-units', dest='h_units', type=int, default=32,
                    help='number of hidden units [32]')
parser.add_argument('--rnn-layers', dest='rnn_layers', type=int, default=3,
                    help='number of layers for rnn [5]')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.3,
                    help='dropout for lstm/tcn layers [0.2]')
parser.add_argument('--kernel-size', dest='kernel_size', type=int, default=5,
                    help='kernel size for tcn conv, use odd ints [5]')

parser.add_argument('--comment', dest='comment', type=str, default='',
                    help='extra comments')
args = parser.parse_args()

# Initialize W&B project
wandb.init(project="SNe_Spec", name=args.run_name)

# save hyper-parameters to W&B
wandb.config.update(args)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


## run main program
def run_code():

    ## Load Data ##
    dataset = DataSet()

    if len(dataset) == 0:
        print('No items in training set...')
        print('Exiting!')
        sys.exit()

    ## data loaders for training and testing
    train_loader, test_loader = dataset.get_dataloader(batch_size=batch_size,
                                                       shuffle=rnd_seed,
                                                       test_split=.2,
                                                       random_seed=rnd_seed)
    print('')
    print('Training lenght: ', len(train_loader) * batch_size)
    print('Test lenght    : ', len(test_loader) * batch_size)

    ## Define GAN model, Ops, and Train ##
    model = ## Load model

    wandb.watch(model, log='gradients')

    print('Summary:')

    # print(generator)
    wandb.config.n_train_params = count_parameters(model)
    print('Num of trainable params: ', count_parameters(vae))
    print('\n')

    # Initialize optimizers
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # Learning Rate scheduler
    if lr_sch == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=20,
                                              gamma=0.5)
    elif lr_sch == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=0.985)
    elif lr_sch == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=50,
                                                         eta_min=1e-5)
    elif lr_sch == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=.5,
                                                         verbose=True)
    else:
        scheduler = None

    print('Optimizer    :', optimizer)
    print('LR Scheduler :', scheduler.__class__.__name__)

    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print('########################################')
    print('########  Running in %4s  #########' % (device))
    print('########################################')

    trainer = Trainer(model, optimizer, batch_size, wandb,
                      print_every=200,
                      device=device,
                      scheduler=scheduler)

    if args.dry_run:
        print('******** DRY RUN ******** ')
        return

    trainer.train(train_loader, test_loader, num_epochs,
                  save=True, early_stop=args.early_stop)


if __name__ == "__main__":
    print('Running in: ', machine, '\n')
    for key, value in vars(args).items():
        print('%15s\t: %s' % (key, value))

    run_code()
