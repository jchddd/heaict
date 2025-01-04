from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader
import warnings
import os 
import random
import copy

warnings.filterwarnings('ignore')

class trainer():
    def __init__(
        self, 
        model, 
        graph_list,
        epoch=100,
        batch=64,
        metric=['mae', 0.9, 5, 3, 1.0],
        optimizer=['AdamW', 0.01, 5e-4],
        scheduler=['ConstantLR'],
    ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # input parameter
        self.classfy = True
        
        self.epoch = epoch
        self.batch = batch
        self.metric = metric
        self.optimizer = optimizer
        self.scheduler = scheduler
        # init data
        self.init_model = model.to(device)
        self.graph_list = [graph.to(device) for graph in graph_list]
        self.datasets = {'train':[], 'valid':[], 'test':[]}
        self.loaders = []
        # mission type
        self.mission_type = ''
        self.number_train = 0
        
        self.cv = False
        self.cv_k = 0
        # train result
        self.model = []
        self.best_score = []
        self.best_weight = []
        self.train_history = {'lr':[], 'train':[], 'valid':[], 'test':[]}
        
    def split_train_valid_test(self, train_ratio=0.6, valid_ration=0.2, test_ration=0.2):
        dataset_train, dataset_valid, dataset_test = random_split(self.graph_list, [train_ratio, valid_ration, test_ration])
        self.datasets['train'].append(dataset_train)
        self.datasets['valid'].append(dataset_valid)
        self.datasets['test'].append(dataset_test)
        self.number_train = 1
        self.mission_type = 'normal'
    
    def split_K_fold(self, K=5):
        self.cv = True; self.cv_k = K
        self.number_train = K
        self.mission_type = 'cv'
        
        kfold = KFold(n_splits=K, shuffle=True)
        for train_index, valid_index in kfold.split(self.graph_list):
            self.datasets['train'].append(Subset(self.graph_list, train_index))
            self.datasets['valid'].append(Subset(self.graph_list, valid_index))
            self.datasets['test'].append(None)

    def train(self, print_epoch=True, print_interval=1, use_train_loss=False):
        self._load_loader()
        
        for number_train in range(self.number_train):                
            print('Model %2d - start training:' % (number_train)) if print_epoch else None
            print('epoch | loss_train | loss_valid | loss_test')  if print_epoch else None
            self._init_train()
            
            metric = self._load_metric()
            optimizer = self._load_optimizer(self.model[number_train])
            scheduler = self._load_scheduler(optimizer)
            loader_train, loader_valid, loader_test = self.loaders[number_train]
            
            for epoch in range(self.epoch):
                loss_train = self._train(self.model[number_train], loader_train, optimizer, metric)
                loss_valid = self._evaluate(self.model[number_train], loader_valid, metric)
                loss_test = self._evaluate(self.model[number_train], loader_test, metric)
                
                score = loss_valid if loss_valid != 0 and not use_train_loss else loss_train
                if score < self.best_score[number_train]:
                    self.best_score[number_train] = score
                    self.best_weight[number_train] = copy.deepcopy(self.model[number_train].state_dict())
                
                if self.scheduler[0] == 'ReduceLROnPlateau':
                    scheduler.step(score)
                else:
                    scheduler.step()
                current_lr = scheduler.get_last_lr()[0] if epoch != 0 else self.optimizer[1]
                
                self.train_history['lr'][number_train].append(current_lr)
                self.train_history['train'][number_train].append(loss_train)
                self.train_history['valid'][number_train].append(loss_valid)
                self.train_history['test'][number_train].append(loss_test)
                
                if print_epoch and (epoch == 0 or epoch == self.epoch - 1 or (epoch + 1) % print_interval == 0):
                    print('%5d | %10.6f | %10.6f | %10.6f' % (epoch, loss_train, loss_valid, loss_test))
                
        self._restore()
    
    def show_result(self, model_index=0):
        # init figure
        plt.figure(figsize=(6, 6))
        plt.xlabel('true value')
        plt.ylabel('predict value')
        plt.plot([-10, 10], [-10, 10], zorder=3)
        # plt.axis([-4, 4, -4, 4])
        # init
        loader_type = ['train:', 'valid', 'test']
        colors = ['#ffff80', '#0080ff', '#ff0080']
        # loop dataloader
        maxv = []; minv = []
        for i, loader_name in enumerate(loader_type):
            # get yp
            loader = self.loaders[model_index][i]
            if loader is not None:
                target, predict = self._get_target_predict(self.model[model_index], loader)
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
    
    def show_learn_curve(self, model_index=0):
        x  = np.arange(0, self.epoch) + 1
        y1 = self.train_history['train'][model_index]
        y2 = self.train_history['lr'][model_index]

        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(x, y1, 'r-')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('train loss')

        ax2 = ax1.twinx()
        ax2.plot(x, y2, 'b-')
        ax2.set_ylabel('learning rate')

        plt.show()
        
    def calculate_metrics(self):
        loader_type = {0: 'train', 1: 'valid', 2: ' test'}
        for i, model in enumerate(self.model):
            print(f'---------------------------- Model  {i} ----------------------------')
            for j, loader in enumerate(self.loaders[i]):
                if loader is not None:
                    y, yp = self._get_target_predict(model, loader)
                    mae = mean_absolute_error(yp[0], y[0])
                    mse = mean_squared_error(yp[0], y[0])
                    r2s = r2_score(y[0], yp[0])
                    print(f'=== {loader_type[j]} ===') 
                    print(f'MAE:      {mae: .3f} | MSE:       {mse: .3f} | RMSE:   {np.sqrt(mse): .3f} | R2: {r2s: .3f}')
                    if model.classify:
                        for k in [1, 2]:
                            acr = accuracy_score(y[k], yp[k])
                            prc = precision_score(y[k], yp[k], average=None)
                            rec = recall_score(y[k], yp[k], average=None)
                            f1s = f1_score(y[k], yp[k], average=None)
                            sprc = ' '.join([str(round(p, 2)) for p in prc])
                            srec = ' '.join([str(round(r, 2)) for r in rec])
                            print(f'Accuracy: {acr: .3f} | Precision: {np.mean(prc): .3f} | Recall: {np.mean(rec): .3f} | F1: {np.mean(f1s): .3f} | Precisions: {sprc} | Recalls: {srec}')

    def _get_target_predict(self, model, loader):
        '''
        Input a model and a dataloader, return two array for all target values and predicted values.
        '''
        # init
        target = []; predict = []
        true_energies    = []; true_class1    = []; true_class2    = []
        predict_energies = []; predict_class1 = []; predict_class2 = []
        # predict
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
        for number_train in range(self.number_train):
            self.loaders.append([])
            for data_type in ['train', 'valid', 'test']:
                dataset = self.datasets[data_type][number_train]
                if dataset is not None and len(dataset) > 0:
                    self.loaders[number_train].append(DataLoader(dataset, batch_size=self.batch, shuffle=True))
                else:
                    self.loaders[number_train].append(None)
    
    def _init_train(self):
        self.model.append(copy.deepcopy(self.init_model))
        self.best_score.append(666)
        self.best_weight.append(None)
        for key in self.train_history.keys():
            self.train_history[key].append([])
    
    def _train(self, model, loader, optimizer, metric):
        metrics = []
        
        model.train()
        for data in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = metric(output, data.y)
            loss.backward()
            optimizer.step()
            metrics.append(loss.detach().item())
        return np.mean(metrics)

    def _evaluate(self, model, loader, metric):
        if loader is not None:
            metrics = []

            model.eval()
            with torch.no_grad():
                for data in loader:
                    output = model(data)
                    loss = metric(output, data.y)
                    metrics.append(loss.detach().item())
            return np.mean(metrics)
        else:
            return 0
        
    def _restore(self):
        for number_train in range(self.number_train):
            if self.best_weight[number_train] != None:
                self.model[number_train].load_state_dict(self.best_weight[number_train])


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, yp, y):
        mse = self.mse_loss(yp, y)
        rmse = torch.sqrt(mse)
        return rmse



class hybrid_loss(nn.Module):
    def __init__(self, regression_metric='mae', weight_regression=0.5, num_adsb_class=5, num_site_class=3, beta=1.0):
        super().__init__()
        '''
        
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_regression      = torch.Tensor([weight_regression]).to(self.device)
        self.weight_classification  = torch.Tensor([1 - weight_regression]).to(self.device)
        
        if   regression_metric == 'mae':
            self.metric_energy      = nn.SmoothL1Loss(beta=beta)
        elif regression_metric == 'mse':
            self.metric_energy      = nn.MSELoss()
        self.metric_adsb            = nn.CrossEntropyLoss()
        self.metric_site            = nn.CrossEntropyLoss()

    def forward(self, yp, y):
        loss_energy = self.metric_energy(yp[0], y[:, 0].unsqueeze(dim=-1))
        loss_adsb   = self.metric_adsb(yp[1], y[:, 1].long().squeeze())
        loss_site   = self.metric_site(yp[2], y[:, 2].long().squeeze())
        loss        = self.weight_regression * loss_energy + self.weight_classification * (loss_adsb + loss_site)
        return loss

    
def fix_seed(seed):
    '''
    Function used to set the random seed

    Args:
        - seed: The random seed / int
    '''
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
