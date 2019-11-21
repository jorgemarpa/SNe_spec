import os
import numpy as np
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.training_callbacks import EarlyStopping
from src.utils import *
from sklearn import metrics


class Trainer_Class():
    def __init__(self, model, optimizer, batch_size, wandb,
                 scheduler=None, print_every=50,
                 label_enc=None,
                 device='cpu'):

        self.device = device
        self.model = model
        if torch.cuda.device_count() > 1 and True:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        print('Is model in cuda? ',
              next(self.model.parameters()).is_cuda)
        self.opt = optimizer
        self.sch = scheduler
        self.batch_size = batch_size
        self.train_loss = {'Loss': []}
        self.val_loss = {'Loss': [], 'F1': [], 'Acc': []}
        self.num_steps = 0
        self.print_every = print_every
        self.loss = nn.NLLLoss(reduction='sum')  # test with reduction=sum
        self.wb = wandb
        self.l_enc = label_enc


    def _loss(self, y, yhat, train=True):
        loss = self.loss(yhat, y)

        if train:
            self.train_loss['Loss'].append(loss.item() / len(y))
        else:
            self.val_loss['Loss'].append(loss.item() / len(y))

        return loss


    def _metrics(self, y, yhat, cm=False):
        pred = np.argmax(yhat, axis=1)
        f1 = metrics.f1_score(y, pred, average='weighted')
        self.val_loss['F1'].append(f1)
        acc = metrics.accuracy_score(y, pred)
        self.val_loss['Acc'].append(acc)

        if cm:
            conf_m = metrics.confusion_matrix(y, pred,
                                              labels=None)
            fig, img = plot_conf_matrix(conf_m, set(y),
                                        cl_names=self.l_enc.classes_)

            self.wb.log({'Conf_Matrix': self.wb.Image(fig)},
                        step=self.num_steps)


    ## function that does the in-epoch training
    def _train_epoch(self, data_loader, epoch):
        self.model.train()
        ## iterate over len(x)/batch_size
        for i, (x, _, y) in enumerate(data_loader):

            self.num_steps += 1
            x = x.to(self.device)
            y = y.to(self.device)
            yhat = self.model(x)

            loss = self._loss(y, yhat, train=True)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            self._report(i, train=True)


    def _val_epoch(self, test_loader, epoch):
        self.model.eval()
        y_all, yhat_all = [], []
        with torch.no_grad():
            for i, (x, _, y) in enumerate(test_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                y_all.extend(y.cpu().detach().numpy())

                yhat = self.model(x)
                yhat_all.extend(yhat.cpu().detach().numpy())

                loss = self._loss(y, yhat, train=False)

        self._metrics(np.array(y_all),
                      np.array(yhat_all),
                      cm=True if epoch % 5 else False)
        self._report(i, train=False, force=True)

        return loss



    def train(self, train_loader, test_loader, epochs,
              save=True, early_stop=False):

        ## hold samples, real and generated, for initial plotting
        if early_stop:
            print('-----------> Early Stop ON <------------')
            early_stopping = EarlyStopping(patience=10, min_delta=.02,
                                           verbose=True)

        ## train for n number of epochs
        time_start = datetime.datetime.now()
        for epoch in range(1, epochs + 1):
            e_time = datetime.datetime.now()
            print('##'*20)
            print("\nEpoch {}".format(epoch))

            # train and validate
            self._train_epoch(train_loader, epoch)
            val_loss_ = self._val_epoch(test_loader, epoch)

            # update learning rate according to scheduler
            if self.sch is not None:
                self.wb.log({'LR': self.opt.param_groups[0]['lr']},
                            step=self.num_steps)
                if 'ReduceLROnPlateau' == self.sch.__class__.__name__:
                    self.sch.step(val_loss_)
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
                early_stopping(val_loss_.cpu())
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        if save:
            torch.save(self.model.state_dict(),
                       '%s/model_%s.pt' %
                       (self.wb.run.dir, self.wb.run.name))

    def test_in(self, test_spec, test_y):

        print('')
        print('########################################')
        print('###########      TEST    ###############')
        print('########################################')
        print('')

        test_spec = torch.from_numpy(test_spec).to(self.device)
        test_label = torch.from_numpy(test_y).to(self.device)
        test_pred_lp = self.model(test_spec)
        test_pred_p = torch.exp(test_pred_lp).cpu().detach().numpy()
        test_pred_y = np.argmax(test_pred_p, axis=1)

        loss = self.loss(test_pred_lp, test_label)
        self.wb.run.summary['Hold-out_Loss'] = loss.item() / len(test_label)
        f1 = metrics.f1_score(test_y, test_pred_y, average='weighted')
        self.wb.run.summary['Hold-out_F1'] = f1
        acc = metrics.accuracy_score(test_y, test_pred_y)
        self.wb.run.summary['Hold-out_Acc'] = acc

        cm = metrics.confusion_matrix(test_y, test_pred_y,
                                          labels=None)
        fig, img = plot_conf_matrix(cm, set(test_y),
                                    cl_names=self.l_enc.classes_)

        self.wb.log({'Hold-out_Conf_Matrix': self.wb.Image(fig)})

        print('Hold-out Loss: %.4f' % loss)
        print('Hold-out F1  : %.4f' % f1)
        print('Hold-out Acc : %.4f' % acc)
        print("__"*20)



    def _report(self, i, train=True, force=False):
        ## ------------------------ Reports ---------------------------- ##
        ## print scalars to std output and save scalars/hist to W&B
        if i % self.print_every == 0 or force:
            if train:
                print("Training iteration %i, global step %i" % (i + 1, self.num_steps))
                print("Loss: %3.4f" % (self.train_loss['Loss'][-1]))
                self.wb.log({'Train_Loss': self.train_loss['Loss'][-1]},
                            step=self.num_steps)
            else:
                print('*** VALIDATION ***')
                print("Loss: %.4f" % (self.val_loss['Loss'][-1]))
                self.wb.log({'Val_Loss': self.val_loss['Loss'][-1]},
                            step=self.num_steps)
                self.wb.log({'Val_F1': self.val_loss['F1'][-1]},
                            step=self.num_steps)
                self.wb.log({'Val_Acc': self.val_loss['Acc'][-1]},
                            step=self.num_steps)



            print("--"*20)



############################################################################
############################   Regressor   #################################
############################################################################
class Trainer_Regr():
    def __init__(self, model, optimizer, batch_size, wandb,
                 scheduler=None, print_every=50,
                 device='cpu'):

        self.device = device
        self.model = model
        if torch.cuda.device_count() > 1 and True:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        print('Is model in cuda? ',
              next(self.model.parameters()).is_cuda)
        self.opt = optimizer
        self.sch = scheduler
        self.batch_size = batch_size
        self.train_loss = {'Loss': [],
                          'Loss_phase': [],
                          'Loss_dm15': []}
        self.val_loss = {'Loss': [],
                         'Loss_phase': [],
                         'Loss_dm15': []}
        self.num_steps = 0
        self.print_every = print_every
        self.loss = nn.MSELoss(reduction='sum')  # test with reduction=sum
        self.wb = wandb


    ## loss function
    def _loss(self, y, phat, dmhat, train=True):
        loss_p = self.loss(phat, y[:,0].unsqueeze(-1))
        loss_dm = self.loss(dmhat, y[:,1].unsqueeze(-1))

        if train:
            self.train_loss['Loss_phase'].append(loss_p.item() / len(y))
            self.train_loss['Loss_dm15'].append(loss_dm.item() / len(y))
            self.train_loss['Loss'].append((loss_dm.item() + loss_p.item())
                                           / len(y))
        else:
            self.val_loss['Loss_phase'].append(loss_p.item() / len(y))
            self.val_loss['Loss_dm15'].append(loss_dm.item() / len(y))
            self.val_loss['Loss'].append((loss_dm.item() + loss_p.item())
                                         / len(y))

        return loss_p + loss_dm


    ## function that does the in-epoch training
    def _train_epoch(self, data_loader, epoch):
        self.model.train()
        ## iterate over len(x)/batch_size
        for i, (x, y) in enumerate(data_loader):

            self.num_steps += 1
            x = x.to(self.device)
            y = y.to(self.device)
            phat, dmhat = self.model(x)

            loss = self._loss(y, phat, dmhat, train=True)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            self._report(i, train=True)


    ## test in validation dataset
    def _val_epoch(self, test_loader, epoch):
        self.model.eval()
        y_all, phat_all, dmhat_all = [], [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                y_all.extend(y.cpu().detach().numpy())

                phat, dmhat = self.model(x)
                phat_all.extend(phat.cpu().detach().numpy())
                dmhat_all.extend(dmhat.cpu().detach().numpy())

                loss = self._loss(y, phat, dmhat, train=False)

        self._report(i, train=False, force=True)

        yhat_all = np.array([phat_all, dmhat_all]).squeeze(-1).T
        y_all = np.array(y_all)

        self.wb.log({'Val_RMSE_Phase': rmse(yhat_all[:,0],
                                            y_all[:,0], unscale = 'phase')},
                    step=self.num_steps)
        self.wb.log({'Val_RMSE_dm15' : rmse(yhat_all[:,1],
                                            y_all[:,1], unscale = 'dm15')},
                    step=self.num_steps)
        if epoch % 2 == 0:
            fig = residuals_scatter_plot(y_all,
                                         yhat_all,
                                         epoch=epoch)
            self.wb.log({'Val_Regression': self.wb.Image(fig)})

        return loss



    ## main training loop
    def train(self, train_loader, test_loader, epochs,
              save=True, early_stop=False):

        ## hold samples, real and generated, for initial plotting
        if early_stop:
            print('-----------> Early Stop ON <------------')
            early_stopping = EarlyStopping(patience=10, min_delta=.02,
                                           verbose=True)

        ## train for n number of epochs
        time_start = datetime.datetime.now()
        for epoch in range(1, epochs + 1):
            e_time = datetime.datetime.now()
            print('##'*20)
            print("\nEpoch {}".format(epoch))

            # train and validate
            self._train_epoch(train_loader, epoch)
            val_loss_ = self._val_epoch(test_loader, epoch)

            # update learning rate according to scheduler
            if self.sch is not None:
                self.wb.log({'LR': self.opt.param_groups[0]['lr']},
                            step=self.num_steps)
                if 'ReduceLROnPlateau' == self.sch.__class__.__name__:
                    self.sch.step(val_loss_)
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
                early_stopping(val_loss_.cpu())
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        if save:
            torch.save(self.model.state_dict(),
                       '%s/model_%s.pt' %
                       (self.wb.run.dir, self.wb.run.name))


    ## test in hold-out dataset
    def test_in(self, test_spec, test_y):

        print('')
        print('########################################')
        print('###########      TEST    ###############')
        print('########################################')
        print('')

        test_spec = torch.from_numpy(test_spec).to(self.device)
        test_y = torch.from_numpy(test_y).to(self.device)
        phat, dmhat = self.model(test_spec)

        loss_p = self.loss(phat, test_y[:,0].unsqueeze(-1))
        loss_dm = self.loss(dmhat, test_y[:,1].unsqueeze(-1))
        self.wb.run.summary['Hold-out_Loss_phase'] = loss_p.item() / len(test_y)
        self.wb.run.summary['Hold-out_Loss_dm15'] = loss_dm.item() / len(test_y)
        test_y = test_y.cpu().detach().numpy()
        phat = phat.cpu().detach().numpy()
        dmhat = dmhat.cpu().detach().numpy()
        phase_rms = rmse(phat, test_y[:,0], unscale = 'phase')
        dm15_rms = rmse(dmhat, test_y[:,1], unscale = 'dm15')
        self.wb.log({'Hold-out_RMSE_Phase': phase_rms})
        self.wb.log({'Hold-out_RMSE_dm15' : dm15_rms})

        print('Hold-out Phase RMS: %.4f' % phase_rms)
        print('Hold-out dm15 RMS : %.4f' % dm15_rms)
        print("__"*20)



    ## report metrics
    def _report(self, i, train=True, force=False):
        ## ------------------------ Reports ---------------------------- ##
        ## print scalars to std output and save scalars/hist to W&B
        if i % self.print_every == 0 or force:
            if train:
                print("Training iteration %i, global step %i" %
                      (i + 1, self.num_steps))
                print("Loss: %3.4f" % (self.train_loss['Loss'][-1]))
                self.wb.log({'Train_Loss_phase': self.train_loss['Loss_phase'][-1]},
                            step=self.num_steps)
                self.wb.log({'Train_Loss_dm15': self.train_loss['Loss_dm15'][-1]},
                            step=self.num_steps)
                self.wb.log({'Train_Loss': self.train_loss['Loss'][-1]},
                            step=self.num_steps)
            else:
                print('*** VALIDATION ***')
                print("Loss: %.4f" % (self.val_loss['Loss'][-1]))
                self.wb.log({'Val_Loss_phase': self.val_loss['Loss_phase'][-1]},
                            step=self.num_steps)
                self.wb.log({'Val_Loss_dm15': self.val_loss['Loss_dm15'][-1]},
                            step=self.num_steps)
                self.wb.log({'Val_Loss': self.val_loss['Loss'][-1]},
                            step=self.num_steps)
            print("--"*20)
