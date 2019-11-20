import sys
import datetime
import argparse
import torch
import torch.optim as optim
import numpy as np
from src.dataset import *
from src.trainer import *
from src.model import *
from src.utils import str2bool
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
parser.add_argument('--run-name', dest='run_name', type=str, default='',
                    help='Name of the run')
parser.add_argument('--machine', dest='machine', type=str,
                    default='exalearn',
                    help='Where is running ([local], colab, exalearn)')

parser.add_argument('--mode', dest='mode', type=str, default='clas',
                    help='Whether to do classification or regression ([clas], regr)')
parser.add_argument('--data', dest='data', type=str,
                    default='spec-111519-1696',
                    help='Which data use ([spec])')

parser.add_argument('--optim', dest='optim', type=str, default='adam',
                    help='Optimizer ([adam],sdg)')
parser.add_argument('--lr', dest='lr', type=float, default=1e-3,
                    help='learning rate [1e-3]')
parser.add_argument('--lr-sch', dest='lr_sch', type=str, default=None,
                    help='learning rate shceduler '+
                    '([None], step, exp, cosine, plateau)')
parser.add_argument('--early-stop', dest='early_stop', type=str2bool,
                    nargs='?', const=True,
                    default=True, help='Early stopping, [True]')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=128,
                    help='batch size [64]')
parser.add_argument('--epochs', dest='num_epochs', type=int, default=150,
                    help='total number of training epochs [150]')

parser.add_argument('--arch', dest='arch', type=str, default='lstm',
                    help='Layer architecture ([lstm],gru,rnn,conv)')
parser.add_argument('--hidden-units', dest='h_units', type=int, default=32,
                    help='number of hidden units [32]')
parser.add_argument('--rnn-layers', dest='rnn_layers', type=int, default=3,
                    help='number of layers for rnn [5]')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.3,
                    help='dropout for rnn/conv layers [0.2]')
parser.add_argument('--rnn-bidir', dest='rnn_bidir', type=str2bool,
                    nargs='?', const=True,
                    default=False, help='Bidirectional RNN [False]')
parser.add_argument('--kernel-size', dest='kernel_size', type=int, default=3,
                    help='kernel size for conv, use odd ints [5]')

parser.add_argument('--comment', dest='comment', type=str, default='',
                    help='extra comments')
args = parser.parse_args()

# Initialize W&B project
wandb.init(entity='deep_sne',
           project="SNe_Spec_%s" % (args.mode),
           notes='%s %s' % (args.run_name, args.comment))

# save hyper-parameters to W&B
wandb.config.update(args)
wandb.config.rnd_seed = rnd_seed

# identify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.cuda.empty_cache()

# function that count models trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


## run classification program
def do_classification():

    ## Load Data ##
    dataset = DataSet_Class(machine=args.machine,
                            timestamp=args.data.split('-')[1],
                            length=args.data.split('-')[2])
    dataset.spec_train += .5
    dataset.spec_test += .5

    wandb.config.spec_len = dataset.spec_len
    wandb.config.spec_nfeat = dataset.spec_nfeat
    wandb.config.total_targets = dataset.total_targets

    if len(dataset) == 0:
        print('No items in training set...')
        print('Exiting!')
        sys.exit()

    ## data loaders for training and testing
    train_loader, test_loader = dataset.get_dataloader(batch_size=args.batch_size,
                                                       shuffle=True,
                                                       test_split=.2,
                                                       random_seed=rnd_seed)
    print('')
    print('Training lenght  : ', len(train_loader) * args.batch_size)
    print('Validation lenght: ', len(test_loader) * args.batch_size)

    ## Define GAN model, Ops, and Train ##
    if args.arch in ['lstm','gru','rnn']:
        model = RecNN_Clas(seq_len=dataset.spec_len,
                           n_feats=dataset.spec_nfeat,
                           hidden_dim=args.h_units,
                           n_layers=args.rnn_layers,
                           rnn=args.arch,
                           out_size=dataset.total_targets,
                           dropout=args.dropout,
                           bidir=args.rnn_bidir,
                           device=device)
    elif args.arch == 'conv':
        model = ConvNN_Clas(seq_len=dataset.spec_len,
                            n_feats=dataset.spec_nfeat,
                            kernel_size=args.kernel_size,
                            hidden_channels=args.h_units,
                            out_size=dataset.total_targets,
                            dropout=args.dropout)

    wandb.watch(model, log='all')

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
    print('########################################')
    print('########  Running in %4s  #############' % (device))
    print('########################################')

    trainer = Trainer_Class(model, optimizer, args.batch_size, wandb,
                            print_every=50,
                            device=device,
                            scheduler=scheduler,
                            label_enc=dataset.label_int_enc)

    if args.dry_run:
        print('******** DRY RUN ******** ')
        return

    trainer.train(train_loader, test_loader, args.num_epochs,
                  save=True, early_stop=args.early_stop)

    trainer.test_in(dataset.spec_test,
                    dataset.label_test_int)
    return


# regression program
def do_regression():

    ## Load Data ##
    dataset = DataSet_Regr(machine=args.machine,
                           timestamp=args.data.split('-')[1],
                           length=args.data.split('-')[2])
    dataset.spec_train += .5
    dataset.spec_test += .5

    wandb.config.spec_len = dataset.spec_len
    wandb.config.spec_nfeat = dataset.spec_nfeat
    wandb.config.total_targets = dataset.total_targets


    if len(dataset) == 0:
        print('No items in training set...')
        print('Exiting!')
        sys.exit()

    ## data loaders for training and testing
    train_loader, test_loader = dataset.get_dataloader(batch_size=args.batch_size,
                                                       shuffle=True,
                                                       test_split=.2,
                                                       random_seed=rnd_seed)
    print('')
    print('Training lenght  : ', len(train_loader) * args.batch_size)
    print('Validation lenght: ', len(test_loader) * args.batch_size)

    ## Define GAN model, Ops, and Train ##
    if args.arch in ['lstm','gru','rnn']:
        model = RecNN_Regr(seq_len=dataset.spec_len,
                           n_feats=dataset.spec_nfeat,
                           hidden_dim=args.h_units,
                           n_layers=args.rnn_layers,
                           rnn=args.arch,
                           out_size=dataset.total_targets,
                           dropout=args.dropout,
                           bidir=args.rnn_bidir,
                           device=device)
    elif args.arch == 'conv':
        model = ConvNN_Regr(seq_len=dataset.spec_len,
                            n_feats=dataset.spec_nfeat,
                            kernel_size=args.kernel_size,
                            hidden_channels=args.h_units,
                            out_size=dataset.total_targets,
                            dropout=args.dropout)

    wandb.watch(model, log='all')

    print('Model Summary:')
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
    print('########################################')
    print('########  Running in %4s  #############' % (device))
    print('########################################')

    trainer = Trainer_Regr(model, optimizer, args.batch_size, wandb,
                           scheduler=scheduler,
                           print_every=50,
                           device=device)

    if args.dry_run:
        print('******** DRY RUN ******** ')
        return

    trainer.train(train_loader, test_loader, args.num_epochs,
                  save=True, early_stop=args.early_stop)

    trainer.test_in(dataset.spec_test,
                    dataset.target_test)

    return



if __name__ == "__main__":
    for key, value in vars(args).items():
        print('%15s\t: %s' % (key, value))

    if args.mode in ['clas', 'class', 'classification']:
        do_classification()
    elif args.mode in ['reg', 'regr', 'regre', 'regression']:
        do_regression()
    else:
        print('Wrong mode (%s), please select between [reg, clas]')
        sys.exit()
