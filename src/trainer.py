import os
import numpy as np
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.training_callbacks import EarlyStopping


class Trainer():
    def __init__(self, model, optimizer, batch_size, wandb,
                 scheduler=None, print_every=50,
                 device='cpu'):

        self.device = device
        self.model = model
        if torch.cuda.device_count() > 1 and True:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        print('Is model in cuda? ', next(self.model.parameters()).is_cuda)
        self.opt = optimizer
        self.sch = scheduler
        self.batch_size = batch_size
        self.train_loss = {'Loss': []}
        self.test_loss = {'Loss': []}
        self.num_steps = 0
        self.print_every = print_every
        self.loss = nn.NLLLoss(reduction='sum')
        self.wb = wandb


    def _loss(self, y, yhat, train=True):
        loss = self.loss(yhat, y)

        if train:
            self.train_loss['Loss'].append(loss.item() / len(y))
        else:
            self.test_loss['Loss'].append(loss.item() / len(y))

        return loss


    ## function that does the in-epoch training
    def _train_epoch(self, data_loader, epoch):
        self.model.train()
        ## iterate over len(x)/batch_size
        for i, (x, _, y) in enumerate(data_loader):

            self.num_steps += 1
            x = x.to(self.device)
            y = y.to(self.device)
            self.opt.zero_grad()

            yhat = self.model(x)

            loss = self._loss(y, yhat, train=True)
            loss.backward()
            self.opt.step()

            self._report(i, train=True)


    def _test_epoch(self, test_loader, epoch):
        self.model.eval()
        with torch.no_grad():
            for i, (x, _, y) in enumerate(test_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                yhat = self.model(x)

                loss = self._loss(y, yhat, train=False)

        self._report(i, train=False)

        return loss



    def train(self, train_loader, test_loader, epochs,
              save=True, early_stop=False):

        ## hold samples, real and generated, for initial plotting
        if early_stop:
            early_stopping = EarlyStopping(patience=10, min_delta=.1,
                                           verbose=True)

        ## train for n number of epochs
        time_start = datetime.datetime.now()
        for epoch in range(1, epochs + 1):
            e_time = datetime.datetime.now()
            print('##'*20)
            print("\nEpoch {}".format(epoch))

            # train and validate
            self._train_epoch(train_loader, epoch)
            val_loss = self._test_epoch(test_loader, epoch)

            # update learning rate according to scheduler
            if self.sch is not None:
                self.wb.log({'LR': self.opt.param_groups[0]['lr']},
                            step=self.num_steps)
                if 'ReduceLROnPlateau' == self.sch.__class__.__name__:
                    self.sch.step(val_loss)
                else:
                    self.sch.step(epoch)

            # report elapsed time per epoch and total run tume
            epoch_time = datetime.datetime.now() - e_time
            elap_time = datetime.datetime.now() - time_start
            print('Time per epoch: %i s' % epoch_time.seconds)
            print('Elapsed time  : %.2f m' % (elap_time.seconds/60))
            print('##'*20)

            # early stopping
            if early_stop:
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        if save:
            torch.save(self.model.state_dict(),
                       '%s/model_%s.pt' %
                       (self.wb.run.dir, self.wb.run.name))


    def _report(self, i, train=True):
        ## ------------------------ Reports ---------------------------- ##
        ## print scalars to std output and save scalars/hist to W&B
        if i % self.print_every == 0:
            if train:
                print('*** TRAIN ***')
                print("Iteration %i, global step %i" % (i + 1, self.num_steps))
                print("Loss: %3.2f" % (self.train_loss['Loss'][-1]))
                self.wb.log({'Train_Loss': self.train_loss['Loss'][-1]},
                            step=self.num_steps)
            else:
                print('*** TEST ***')
                print("Epoch %i, global step %i" % (ep, self.num_steps))
                print("Loss: %.2f" % (self.test_loss['Loss'][-1]))
                self.wb.log({'Test_Loss'     : self.test_loss['Loss'][-1]},
                            step=self.num_steps)



            print("__"*20)
