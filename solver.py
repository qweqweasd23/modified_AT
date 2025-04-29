import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment
import contextlib
import pandas as pd
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext
import torch.profiler
from tqdm import tqdm

def my_kl_loss(p, q):
    p = torch.clamp(p, min=1e-7)
    q = torch.clamp(q, min=1e-7)
    return torch.mean(torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1), dim=1)


def adjust_learning_rate(optimizer, decay_step, base_lr):
    decay_factor = 0.7 ** decay_step
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'prior':
            param_group['lr'] = base_lr * 10 * decay_factor
        else:
            param_group['lr'] = base_lr * decay_factor
    print(f'Updated LRs - Series: {base_lr * decay_factor:.2e}, Prior: {base_lr * 10 * decay_factor:.2e}')

def window_to_original(scores, win_size, original_length):
    aggregated = np.zeros(original_length)
    counts = np.zeros(original_length)
    
    for i in range(len(scores)):
        start = i
        end = i + win_size
        aggregated[start:end] += scores[i]
        counts[start:end] += 1
    
    aggregated /= np.where(counts == 0, 1, counts)  # 평균 계산
    return aggregated


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss,val_loss2, model, path):
        score = val_loss
        score2 = val_loss2

        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)

        elif ((score - self.best_score + self.delta) + (score2 - self.best_score2 + self.delta)) < 0:
            print(f'score changed (loss1 증가량 + loss2 감소량): {-((score - self.best_score + self.delta) + (score2 - self.best_score2 + self.delta)):.6f} ')
            self.counter = 0
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            

        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss,val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss1 change  : {self.val_loss_min:.6f} --> {val_loss:.6f}.')
            print(f'Validation loss2 change  : {self.val_loss2_min:.6f} --> {val_loss2:.6f}.')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset,L=self.L)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset,L=self.L)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset,L=self.L)


        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.decay_step = 0  

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)

        self.optimizer=torch.optim.Adam([
                        {'params': self.model.series_params, 'lr': self.lr, 'name': 'series'},
                        {'params': self.model.other_params, 'lr': self.lr, 'name': 'other'},
                        {'params': self.model.prior_params, 'lr': self.lr*10, 'name': 'prior'}
                    ])

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        rec_losses = []
        with torch.no_grad():  
            for i, (input_data, _) in enumerate(vali_loader):
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)
                rec_loss = self.criterion(output, input)
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size))))
                    prior_loss += torch.mean(my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u]))
                    
                series_loss = rec_loss - self.k * (series_loss/ len(prior))
                prior_loss = rec_loss + self.k * (prior_loss / len(prior))

                #rec_loss = self.criterion(output, input)
                rec_losses.append(rec_loss.item())

                loss_1.append(series_loss.item())
                loss_2.append(prior_loss.item())
            return np.average(loss_1), np.average(loss_2), np.average(rec_losses)
    def profiler_context(self):
        if self.enable_profiler:
            return torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{self.dataset}'),
                record_shapes=True,
                with_stack=True
            )
        else:
            return nullcontext()
    def train(self):

        print("======================TRAIN MODE======================")
        debug_mode = False
        self.enable_profiler = False  
        #self.train()
        with torch.autograd.detect_anomaly() if debug_mode else contextlib.nullcontext():
            time_now = time.time()
            path = self.model_save_path
            if not os.path.exists(path):
                os.makedirs(path)
            early_stopping = EarlyStopping(patience=10, verbose=True, dataset_name=self.dataset)
            train_steps = len(self.train_loader)
            with self.profiler_context() as prof:
                for epoch in range(self.num_epochs):
                    iter_count = 0
                    loss1_list = []
                    loss2_list = []
                    epoch_time = time.time()
                    self.model.train()

                    for i, (input_data, labels) in enumerate(self.train_loader):


                        iter_count += 1
                        input = input_data.float().to(self.device)

                        

                        # calculate Association discrepancy
                        series_loss = 0.0
                        prior_loss = 0.0

                        # freeze the prior parameters
                        for param in list(self.model.series_params)+ list(self.model.other_params) :
                            param.requires_grad = True
                        for param in list(self.model.prior_params):#+ list(self.model.other_params):
                            param.requires_grad = False
                        output, series, prior, _ = self.model(input)
                        rec_loss = self.criterion(output, input)
                        self.optimizer.zero_grad()
                        
                        
                        for u in range(len(prior)):
                            series_loss += torch.mean(my_kl_loss(series[u], (
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.win_size))))
                        series_loss = series_loss / len(prior)    
                        loss1 = rec_loss - self.k * series_loss

                        loss1.backward()

                        self.optimizer.step()
    
                    
                        
                        
                        # freeze the series parameters
                        for param in list(self.model.series_params)+ list(self.model.other_params):
                            param.requires_grad = False
                        for param in list(self.model.prior_params):#+ list(self.model.other_params):
                            param.requires_grad = True

                        output, series, prior, _ = self.model(input)
                        rec_loss = self.criterion(output, input)
                        self.optimizer.zero_grad()
                        
                        
                        for u in range(len(prior)):    
                            prior_loss +=  torch.mean(my_kl_loss(
                                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.win_size)),
                                series[u]))
                        prior_loss = prior_loss / len(prior)
                        loss2 = rec_loss + self.k * prior_loss

                        loss2.backward()

                        self.optimizer.step()                
     
                        loss1_list.append(series_loss.item())
                        loss2_list.append(prior_loss.item())

                        



                        if self.enable_profiler:
                            prof.step()


                    print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                    train_loss1 = np.average(loss1_list)
                    train_loss2 = np.average(loss2_list)


                    vali_loss1, vali_loss2, val_rec_loss = self.vali(self.test_loader)

                    print(
                        "Epoch: {0}, Steps: {1} | Train Loss1: {2:.7f} Train Loss2: {3:.7f} Vali Loss1: {4:.7f} Vali Loss2: {5:.7f} val_rec_loss : {6:.7f}".format(
                            epoch + 1, train_steps, train_loss1,train_loss2, vali_loss1,vali_loss2, val_rec_loss))
                    early_stopping(vali_loss1,vali_loss2, self.model, path)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
                    if early_stopping.counter != 0 and early_stopping.counter % 3 == 0:
                        self.decay_step += 1
                        adjust_learning_rate(self.optimizer, self.decay_step, self.lr)



    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth'),weights_only=True))
        self.model.eval()
        temperature = 3

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        attens_energy = []
        attens_energy2 = []
        with torch.no_grad():
            for i, (input_data, labels) in enumerate(self.train_loader):
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)
                loss = torch.mean(criterion(input, output), dim=-1)
                series_loss = 0.0
                prior_loss = 0.0
                # metric 2 = S + P
                S_list, P_list = [], []
                for u in range(len(prior)):
                    P_normalized = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
                    if u == 0:
                        series_loss = my_kl_loss(series[u], P_normalized) * temperature
                        prior_loss = my_kl_loss(P_normalized, series[u]) * temperature
                        S_list.append(series[u]* temperature)
                        P_list.append(P_normalized* temperature)               
                    else:
                        series_loss += my_kl_loss(series[u], P_normalized) * temperature
                        prior_loss += my_kl_loss( P_normalized, series[u]) * temperature
                        S_list.append(series[u]* temperature)
                        P_list.append(P_normalized* temperature)                          

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                S_mean=                torch.mean(torch.stack(S_list), dim=0)
                P_mean=                torch.mean(torch.stack(P_list), dim=0)
                metric2=S_mean+P_mean 
                metric2 = metric2.mean(dim=1)    # head 평균 (batch=256, window=100, window=100)
                metric2 = metric2.mean(dim=-1)   # window 축 평균 (batch=256, window=100)
                metric2 = torch.softmax(metric2, dim=-1)   # softmax (batch=256, window=100)                #print("series_loss shape:", series_loss.shape)
                #print("metric2 shape:", metric2.shape)
                #print("loss shape:", loss.shape)
                #print("metric shape:", metric.shape)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                cri = np.mean(cri, axis=-1)
                attens_energy.append(cri)
                cri2=metric2 * loss
                cri2 = cri2.detach().cpu().numpy()
                cri2 = np.mean(cri2, axis=-1)
                attens_energy2.append(cri2)
            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            train_energy = np.array(attens_energy)
            attens_energy2 = np.concatenate(attens_energy2, axis=0).reshape(-1)
            train_energy2= np.array(attens_energy2)

            combined_energy = train_energy
            thresh = np.percentile(combined_energy, 100-self.anormly_ratio)
            thresh2=np.percentile(train_energy2, 100-self.anormly_ratio)
            print("Threshold2 :", thresh2)
            print("Threshold :", thresh)

            # (3) evaluation on the test set
            test_labels = []
            attens_energy = []
            attens_energy2=[]
            for i, (input_data, labels) in enumerate(self.test_loader):
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)

                loss = torch.mean(criterion(input, output), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0
                # mertic 2 = S + P 
                S_list, P_list = [], []
                for u in range(len(prior)):
                    P_normalized = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
                    if u == 0:
                        series_loss = my_kl_loss(series[u], P_normalized) * temperature
                        prior_loss = my_kl_loss(P_normalized, series[u]) * temperature
                        S_list.append(series[u]* temperature)
                        P_list.append(P_normalized* temperature)               
                    else:
                        series_loss += my_kl_loss(series[u], P_normalized) * temperature
                        prior_loss += my_kl_loss( P_normalized, series[u]) * temperature
                        S_list.append(series[u]* temperature)
                        P_list.append(P_normalized* temperature)  
                metric = torch.softmax((-series_loss-prior_loss), dim=-1)
                S_mean=                torch.mean(torch.stack(S_list), dim=0)
                P_mean=                torch.mean(torch.stack(P_list), dim=0)
                metric2=S_mean+P_mean              
                metric2 = metric2.mean(dim=1)    # head 평균 (batch=256, window=100, window=100)
                metric2 = metric2.mean(dim=-1)   # window 축 평균 (batch=256, window=100)
                metric2 = torch.softmax(metric2, dim=-1)   # softmax (batch=256, window=100)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                cri = np.mean(cri, axis=-1)
                cri2= metric2 * loss
                cri2 = cri2.detach().cpu().numpy()
                cri2 = np.mean(cri2, axis=-1)
                attens_energy.append(cri)
                attens_energy2.append(cri2)
            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            attens_energy2 = np.concatenate(attens_energy2, axis=0).reshape(-1)
            test_energy = np.array(attens_energy)
            test_energy2 = np.array(attens_energy2)
            try:
                test_labels_path= os.path.join(self.data_path, str(self.dataset) + '_test_label.npy')
                test_labels = np.load(test_labels_path)
            except(FileNotFoundError, OSError):
                test_labels_path= os.path.join(self.data_path, 'test_label.csv')
                test_labels = np.nan_to_num(pd.read_csv(test_labels_path).iloc[:, 1:])
            test_energy = window_to_original(test_energy, win_size=100, original_length=test_labels.shape[0])
            test_energy2 = window_to_original(test_energy2, win_size=100, original_length=test_labels.shape[0])

            
            pred = (test_energy > thresh).astype(int)
            pred2 = (test_energy2 > thresh2).astype(int)
            np.save(os.path.join(self.model_save_path, str(self.dataset) + 'test_energy.npy'), test_energy)
            np.save(os.path.join(self.model_save_path, str(self.dataset) + 'test_energy2.npy'), test_energy2)


            gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("pred2:   ", pred2.shape)
        print("gt:     ", gt.shape)


        pred = np.array(pred)
        pred2 = np.array(pred2)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        # ROC
        fpr, tpr, _ = roc_curve(gt, test_energy)
        roc_auc = auc(fpr, tpr)

        # PR
        pre, rec, _ = precision_recall_curve(gt, test_energy)
        pr_auc = auc(rec, pre)

        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, roc_auc : {:0.4f}, pr_acu : {:0.4f}".format(
                accuracy, precision,
                recall, f_score,
                roc_auc, pr_auc))
        
        accuracy2 = accuracy_score(gt, pred2)
        precision2, recall2, f_score2, support2 = precision_recall_fscore_support(gt, pred2,
                                                                              average='binary')
        # ROC
        fpr2, tpr2, _ = roc_curve(gt, test_energy2)
        roc_auc2 = auc(fpr2, tpr2)
        # PR
        pre2, rec2, _ = precision_recall_curve(gt, test_energy2)
        pr_auc2 = auc(rec2, pre2)
        print(
            "Accuracy2 : {:0.4f}, Precision2 : {:0.4f}, Recall2 : {:0.4f}, F-score2 : {:0.4f}, roc_auc2 : {:0.4f}, pr_acu2 : {:0.4f}".format(
                accuracy2, precision2,
                recall2, f_score2,
                roc_auc2, pr_auc2))
        
        return accuracy, precision, recall, f_score, roc_auc, pr_auc, accuracy2, precision2, recall2, f_score2, roc_auc2, pr_auc2

