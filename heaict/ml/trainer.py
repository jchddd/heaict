from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import warnings
import copy
import os

warnings.filterwarnings('ignore')


class trainer():
    '''
    Trainer for model training

    Methods:
        - use_predefine_graphlist: Use pre-defined graph dataset.
        - split_train_valid_test: Split train, valid and test dataset according to ration randomly.
        - train: Train model.
        - show_result: Plot true and predicted value Parity plot.
        - show_learn_curve: Plot epoch vs losses and learning rate.
        - calculate_metrics: Calculate metrics.
        - get_target_predict: Input a model and a dataloader, return two array for all target values and predicted values.
        - get_predict: Input a model and a dataloader, return predicted values.
        - save_model
        - load_model
    '''
    def __init__(
        self, 
        model=None, 
        graph_list=[],
        epoch=100,
        batch=64,
        metric=['mae', 0.9, 5, 3, 1.0],
        optimizer=['AdamW', 0.01, 5e-4],
        scheduler=['ConstantLR'],
        max_norm=10.0,
    ):
        '''
        Parameters:
            - model (torch.Modules)
            - graph_list (list of torch_geometric.data): can be an empty list if you wang to using pre-defined data. Default = []
            - epoch (int): training epoch. Default = 100
            - batch (int): batch size. Default = 64
            - metric (list or str): list of metric parameters. Default = ['mae', 0.9, 5, 3, 1.0]
              choose from 'mse', 'rmse', ['mae', (beta)] (SmoothL1), [*para_for_hyb_loss]
            - optimizer ((3, ) list): list of optimizer parameters. [name in torch.optim, lr, weight_decay]. Default = ['AdamW', 0.01, 5e-4]
            - scheduler (list): list of scheduler parameters. [name in torch.optim.lr_scheduler, *other_para]. Default = ['ConstantLR']
            - max_norm (float): clip grad norm. Default = 1.0
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # input parameter        
        self.epoch = epoch
        self.batch = batch
        self.metric = metric
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_norm = max_norm
        # init data
        self.init_model = model.to(self.device)
        self.graph_list = [graph.to(self.device) for graph in graph_list]
        self.datasets = {'train': None, 'valid': None, 'test': None}
        self.loaders = {'train': None, 'valid': None, 'test': None}
        # train result
        self.model = None
        self.best_score = 0
        self.best_weight = None
        self.train_history = {'lr':[], 'train':[], 'valid':[], 'test':[]}

    def use_predefine_graphlist(self, train_graphs, valid_graphs=[], test_graphs=[]):
        '''
        Use pre-defined graph dataset

        Parameters:
            - train_graphs (list): list of torch_geometric.Graph
            - valid_graphs (list): list of torch_geometric.Graph. Default = []
            - test_graphs (list): list of torch_geometric.Graph. Default = []
        '''
        self.datasets['train'] = [graph.to(self.device) for graph in train_graphs]
        self.datasets['valid'] = [graph.to(self.device) for graph in valid_graphs]
        self.datasets['test']  = [graph.to(self.device) for graph in test_graphs]
        
        self._load_loader()
    
    def split_train_valid_test(self, train_ration=0.6, valid_ration=0.2, test_ration=0.2):
        '''
        Split train, valid and test dataset according to ration randomly

        Parameters:
            - train_ration (float). Default = 0.2
            - valid_ration (float). Default = 0.2
            - test_ration (float). Default = 0.2
        '''
        dataset_train, dataset_valid, dataset_test = random_split(self.graph_list, [train_ration, valid_ration, test_ration])
        self.datasets['train'] = dataset_train
        self.datasets['valid'] = dataset_valid
        self.datasets['test']  = dataset_test

        self._load_loader()

    def train(self, print_info=True, print_interval=1, use_train_loss=False, early_stop=False, patience=20):
        '''
        Train model

        Parameters:
            - print_info (bool): Print training information or not. Default = True
            - print_interval (int): Print interval of epochs. Default = 1
            - use_train_loss (bool): Using training set loss as target instead of that of validation set. Default = False
            - early_stop (bool): Open early stop function or not. Default = False
            - patience (int): Early stop patience. Default = 20
        '''
        # init
        print('start training:') if print_info else None
        print('epoch | loss_train | loss_valid | loss_test')  if print_info else None
        self._init_train()
        self.model.softmaxC = False
        metric = self._load_metric()
        optimizer = self._load_optimizer(self.model)
        scheduler = self._load_scheduler(optimizer)
        loader_train = self.loaders['train']
        loader_valid = self.loaders['valid']
        loader_test = self.loaders['test']
        count = 0
        # start loop
        for epoch in range(self.epoch):
            torch.cuda.empty_cache()
            # get loss and gradient
            loss_train = self._train(self.model, loader_train, optimizer, metric, scheduler)
            loss_valid = self._evaluate(self.model, loader_valid, metric)
            loss_test = self._evaluate(self.model, loader_test, metric)
            current_lr = scheduler.get_last_lr()[0] if epoch != 0 else self.optimizer[1]
            # recored
            self.train_history['lr'].append(current_lr)
            self.train_history['train'].append(loss_train)
            self.train_history['valid'].append(loss_valid)
            self.train_history['test'].append(loss_test)
            # print info
            if print_info and (epoch == 0 or epoch == self.epoch - 1 or (epoch + 1) % print_interval == 0):
                print('%5d | %10.6f | %10.6f | %10.6f' % (epoch, loss_train, loss_valid, loss_test))
            # take best score
            score = loss_valid if loss_valid != 0 and not use_train_loss else loss_train
            if score < self.best_score:
                self.best_score = score
                self.best_weight = copy.deepcopy(self.model.state_dict())
                count = 0
            else:
                count +=1
                if count >= patience and early_stop:
                    print(f'Early stopping at epoch {epoch}') if print_info else None
                    break
        # restore best model
        self._restore()
    
    def show_result(self):
        '''
        Plot true and predicted value Parity plot
        '''
        # init figure
        plt.figure(figsize=(6, 6))
        plt.xlabel('true value')
        plt.ylabel('predict value')
        plt.plot([-10, 10], [-10, 10], zorder=3)
        # plt.axis([-4, 4, -4, 4])
        # init
        loader_type = ['train', 'valid', 'test']
        colors = ['#ffff80', '#0080ff', '#ff0080']
        # loop dataloader
        maxv = []; minv = []
        for i, loader_name in enumerate(loader_type):
            # get yp
            loader = self.loaders[loader_name]
            if loader is not None:
                target, predict = self.get_target_predict(self.model, loader)
                y = target[0]
                yp = predict[0]
                maxv.append(max(max(y), max(yp))); minv.append(min(min(y), min(yp)))
                # cal static
                plt.scatter(6, 6, c=colors[i], label=loader_type[i], ec='k',lw=0.1)
                # plot scatter
                plt.scatter(y, yp, c=colors[i], s=6, ec='k',lw=0.1, zorder=6)
        plt.axis([min(minv) * 1.1, max(maxv) * 1.1, min(minv) * 1.1, max(maxv) * 1.1])
        plt.legend()
        plt.show()
    
    def show_learn_curve(self):
        '''
        Plot epoch vs losses and learning rate
        '''
        x  = np.arange(0, self.epoch) + 1
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(x, self.train_history['train'], 'y-', label='train')
        ax1.plot(x, self.train_history['valid'], 'b-', label='valid') if len(self.train_history['valid']) > 0 else None
        ax1.plot(x, self.train_history['test'], 'r-', label='test') if len(self.train_history['test']) > 0 else None
        ax1.legend()
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')

        ax2 = ax1.twinx()
        ax2.plot(x, self.train_history['lr'], 'k-')
        ax2.set_ylabel('learning rate')

        plt.show()
        
    def calculate_metrics(self, print_metric=True, to_dict=False):
        '''
        Calculate metrics

        Parameters:
            - print_metric (bool): print metric info. Default = True
            - to_dict (bool): Return a dict that store each metric valus. Default = False
        Return:
            - a dict with metrics if to_dict
        '''
        metric_dict = {'train': {}, 'valid': {}, 'test': {}} if to_dict else None
        for loader_type in ['train', 'valid', 'test']:
            loader = self.loaders[loader_type]
            if loader is not None:
                y, yp = self.get_target_predict(self.model, loader)
                mae = mean_absolute_error(yp[0], y[0])
                mse = mean_squared_error(yp[0], y[0])
                r2s = r2_score(y[0], yp[0])
                if to_dict: metric_dict[loader_type]['Eads'] = {'mae': mae, 'mse': mse, 'rmse': np.sqrt(mse), 'r2': r2s}
                print(f'=== {loader_type} ===') if print_metric else None
                print(f'MAE:      {mae: .3f} | MSE:       {mse: .3f} | RMSE:   {np.sqrt(mse): .3f} | R2: {r2s: .3f}') if print_metric else None
                if self.model.classify:
                    for k in [1, 2]:
                        acr = accuracy_score(y[k], yp[k])
                        prc = precision_score(y[k], yp[k], average=None)
                        rec = recall_score(y[k], yp[k], average=None)
                        f1s = f1_score(y[k], yp[k], average=None)
                        sprc = ' '.join([str(round(p, 2)) for p in prc])
                        srec = ' '.join([str(round(r, 2)) for r in rec])
                        if to_dict and k == 1: metric_dict[loader_type]['adsb'] = {'acr': acr, 'prc': np.mean(prc), 'rec': np.mean(rec), 'f1': np.mean(f1s)}
                        if to_dict and k == 2: metric_dict[loader_type]['site'] = {'acr': acr, 'prc': np.mean(prc), 'rec': np.mean(rec), 'f1': np.mean(f1s)}
                        print(f'Accuracy: {acr: .3f} | Precision: {np.mean(prc): .3f} | Recall: {np.mean(rec): .3f} | F1: {np.mean(f1s): .3f} | Precisions: {sprc} | Recalls: {srec}') if print_metric else None
        if to_dict:
            return metric_dict

    def get_target_predict(self, model, loader):
        '''
        Input a model and a dataloader, return two array for all target values and predicted values.
        '''
        model.eval()
        model.softmaxC = True
        # init
        target = []; predict = []
        true_energies    = []; true_class1    = []; true_class2    = []
        predict_energies = []; predict_class1 = []; predict_class2 = []
        # predict
        with torch.no_grad():
            for batch in loader:
                predict_data = model(batch)    
                true_energies.append(batch.y[:, 0])
                predict_energies.append(predict_data[0] if model.classify else predict_data) 
                if model.classify:
                    true_class1.append(batch.y[:, 1].long())
                    true_class2.append(batch.y[:, 2].long())
                    predict_class1.append(torch.argmax(predict_data[1], dim=-1))
                    predict_class2.append(torch.argmax(predict_data[2], dim=-1))
        # cat data
        true_energies = torch.cat(true_energies,dim=0).cpu().numpy()
        predict_energies = torch.cat(predict_energies,dim=0).squeeze(-1).cpu().detach().numpy()
        if model.classify:
            true_class1   = torch.cat(true_class1,  dim=0).cpu().numpy()
            true_class2   = torch.cat(true_class2,  dim=0).cpu().numpy()
            predict_class1   = torch.cat(predict_class1  ,dim=0).squeeze(-1).cpu().detach().numpy()
            predict_class2   = torch.cat(predict_class2  ,dim=0).squeeze(-1).cpu().detach().numpy()
        # append
        target.append(true_energies)
        predict.append(predict_energies)
        if model.classify:
            target.append(true_class1)
            target.append(true_class2)
            predict.append(predict_class1)
            predict.append(predict_class2)
        # return
        return target, predict

    def get_predict(self, model, loader):
        '''
        Input a model and a dataloader, return predicted values.
        '''
        model.eval()
        model.softmaxC = True
        # init
        predict = []
        predict_energies = []; predict_class1 = []; predict_class2 = []
        # predict
        with torch.no_grad():
            for batch in loader:
                predict_data = model(batch)    
                predict_energies.append(predict_data[0] if model.classify else predict_data) 
                if model.classify:
                    predict_class1.append(torch.argmax(predict_data[1], dim=-1))
                    predict_class2.append(torch.argmax(predict_data[2], dim=-1))
        # cat data
        predict_energies = torch.cat(predict_energies,dim=0).squeeze(-1).cpu().detach().numpy()
        if model.classify:
            predict_class1   = torch.cat(predict_class1  ,dim=0).squeeze(-1).cpu().detach().numpy()
            predict_class2   = torch.cat(predict_class2  ,dim=0).squeeze(-1).cpu().detach().numpy()
        # append
        predict.append(predict_energies)
        if model.classify:
            predict.append(predict_class1)
            predict.append(predict_class2)
        # return
        return predict

    def save_model(self, path='.', model_name='model'):
        torch.save(self.model, os.path.join(path, model_name + '.pt'))
        torch.save(self.model.state_dict(), os.path.join(path, model_name + '_dict.pt'))

    def load_model(self, file):
        if 'dict' in file:
            self.model = copy.deepcopy(self.init_model)
            self.model.load_state_dict(torch.load(file))
        else:
            self.model = torch.load(file, map_location=self.device)
        
    def _load_metric(self):
        if   self.metric[0] == 'mae' and len(self.metric) in [1, 2]:
            beta = self.metric[1] if len(self.metric) == 2 else 1.0
            return torch.nn.SmoothL1Loss(beta=beta)
        elif self.metric == 'mse':
            return torch.nn.MSELoss()
        elif self.metric == 'rmse':
            return RMSELoss()
        else:
            return hybrid_loss(*self.metric)
    
    def _load_optimizer(self, model):
        if   self.optimizer[0] == 'SGD':
            return torch.optim.SGD(model.parameters(), lr=self.optimizer[1], weight_decay=self.optimizer[2], momentum=0.9)
        else:
            return getattr(torch.optim, self.optimizer[0], None)(model.parameters(), lr=self.optimizer[1], weight_decay=self.optimizer[2])

    def _load_scheduler(self, optimizer):
        return getattr(torch.optim.lr_scheduler, self.scheduler[0], None)(optimizer, *self.scheduler[1:])
    
    def _load_loader(self):
        for dataset_type in ['train', 'valid', 'test']:
            dataset = self.datasets[dataset_type]
            if dataset is not None and len(dataset) > 0:
                if dataset_type == 'train':
                    self.loaders[dataset_type] = DataLoader(dataset, batch_size=self.batch, shuffle=True)
                else:
                    self.loaders[dataset_type] = DataLoader(dataset, batch_size=self.batch, shuffle=False)
    
    def _init_train(self):
        self.model = copy.deepcopy(self.init_model)
        self.best_score = 666
        self.best_weight = None
        for key in self.train_history.keys():
            self.train_history[key] = []
    
    def _train(self, model, loader, optimizer, metric, scheduler):
        metrics = []
        
        model.train()
        for batch in loader:
            output = model(batch)
            loss = metric(output, batch.y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
            optimizer.step()
            metrics.append(loss.detach().item())
            
        if self.scheduler[0] == 'ReduceLROnPlateau':
            scheduler.step(total_loss.detach().item())
        else:
            scheduler.step()
        
        return np.mean(metrics)

    def _evaluate(self, model, loader, metric):
        if loader is not None:
            metrics = []

            model.eval()
            with torch.no_grad():
                for batch in loader:
                    output = model(batch)
                    loss = metric(output, batch.y)
                    metrics.append(loss.detach().item())
            return np.mean(metrics)
        else:
            return 0
        
    def _restore(self):
        if self.best_weight is not None:
            self.model.load_state_dict(self.best_weight)


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, yp, y):
        mse = self.mse_loss(yp, y)
        rmse = torch.sqrt(mse)
        return rmse



class hybrid_loss(nn.Module):
    def __init__(self, regression_metric='mae', weight_regression=0.9, num_adsb_class=5, num_site_class=3, beta=1.0):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_regression      = torch.Tensor([weight_regression]).to(self.device)
        self.weight_classification  = torch.Tensor([1 - weight_regression]).to(self.device)
        
        if   regression_metric == 'mae':
            self.metric_energy      = nn.SmoothL1Loss(beta=beta)
        elif regression_metric == 'mse':
            self.metric_energy      = nn.MSELoss()
        elif regression_metric == 'rmse':
            self.metric_energy      = RMSELoss()
        self.metric_adsb            = nn.CrossEntropyLoss()
        self.metric_site            = nn.CrossEntropyLoss()

    def forward(self, yp, y):
        loss_energy = self.metric_energy(yp[0], y[:, 0].unsqueeze(dim=-1))
        loss_adsb   = self.metric_adsb(yp[1], y[:, 1].long().squeeze())
        loss_site   = self.metric_site(yp[2], y[:, 2].long().squeeze())
        loss        = self.weight_regression * loss_energy + self.weight_classification * (loss_adsb + loss_site)
        return loss