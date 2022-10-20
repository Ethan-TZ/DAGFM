import torch
from torch import nn
from Utils import Config
from Data.dataset import Dataset
from Model import *
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Utils import Logger


class Trainer():
    def __init__(self) -> None:
        config = Config()
        self.ID = config.config_files.split('.')[0]
        self.logger = Logger(config)
        self.interval = config.interval
        self.dataset = Dataset(config)
        self.savedpath = config.savedpath
        self.model = eval(f"{config.model}")(config).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.L2)
        
        self.loss_fn = nn.BCELoss()
        self.best_auc = 0.
        self.epoch = 0
        self.early_stop_cnt = config.early_stop
        self.config = config
        self.dataset.train = self.dataset.train[:int(len(self.dataset.train) * config.train_ratio)]
        self.draw_interval = len(self.dataset.train) // config.draw_loss_points
        if hasattr(config , 'pretrain'):
            self.savedpath = config.pretrain
            self.resume()
    
    @property
    def current_state(self):
        return {
                'optimizer': self.optimizer.state_dict(), 
                'model': self.model.state_dict() , 
                'early_stop_cnt': self.early_stop_cnt , 
                'best_auc':self.best_auc,
                'epoch':self.step
                }
    
    def resume(self):
        save_info = torch.load(self.savedpath)
        self.optimizer.load_state_dict(save_info['optimizer'])
        self.model.load_state_dict(save_info['model'])
        self.epoch = save_info['epoch'] + 1
        self.best_auc = save_info['best_auc']
        self.early_stop_cnt = save_info['early_stop_cnt']
        print("model loaded !")
    
    def run(self):
        self.writer = SummaryWriter(self.config.logdir)
        self.train_process()
        self.evaluation_process()
        self.writer.close()

    def train_process(self):
        for i in range(self.epoch , 1000):
            self.step = i
            self.train_epoch()
            self._valid()        
            torch.save(self.current_state , self.savedpath)
    
    def evaluation_process(self):
        saved_info = torch.load(self.savedpath + '_best')
        self.model.load_state_dict(saved_info['model'])
        auc , logloss = self.test_epoch(self.dataset.test)
        self.logger.record(self.step , auc ,logloss , 'test')
        self.writer.add_scalars('TEST/AUC' , {self.ID : auc} , 0 )
        self.writer.add_scalars('TEST/LOGLOSS' , {self.ID : logloss} , 0)

    def train_epoch(self):
        cnt = 0
        res = 0
        self.model.train()
        for fetch_data in tqdm(self.dataset.train) if self.config.verbose else self.dataset.train:
            cnt += 1
            self.optimizer.zero_grad()
            loss = self.model.calc_loss(fetch_data)
            loss.backward()
            self.optimizer.step()
            res += loss.cpu().item()
            if cnt % self.draw_interval == 0:
                self.writer.add_scalars('TRAIN/LOSS',{self.ID : res / self.draw_interval},self.step * self.config.draw_loss_points + cnt // self.draw_interval)
                res = 0
                    
    def _valid(self):
        auc , logloss = self.test_epoch(self.dataset.val)
        self.logger.record(self.step , auc ,logloss , 'val')
        self.writer.add_scalars('VAL/AUC' , {self.ID : auc} , self.step)
        self.writer.add_scalars('VAL/LOGLOSS' , {self.ID : logloss} , self.step)
        if auc > self.best_auc:
            print('find a better model !')
            self.best_auc = auc
            self.early_stop_cnt = self.config.early_stop
            torch.save(self.current_state , self.savedpath + '_best')
        else:
            self.early_stop_cnt -= 1
        if self.early_stop_cnt == 0:
            self.early_fin()
    
    def early_fin(self):
        self.evaluation_process()
        self.writer.close()
        exit(1)

    def test_epoch(self , datasource):
        with torch.no_grad():
            self.model.eval()
            val , truth = [] , []
            for fetch_data in tqdm(datasource) if self.config.verbose else datasource:
                prediction = self.model(fetch_data)
                val.append(prediction.cpu().numpy())
                truth.append(fetch_data['label'].numpy())

            y_hat = np.concatenate(val, axis=0).squeeze()
            y = np.concatenate(truth, axis=0).squeeze()
            auc = roc_auc_score(y, y_hat)
            logloss = - np.sum(y*np.log(y_hat + 1e-6) + (1-y)*np.log(1-y_hat+1e-6)) /len(y)
        return auc , logloss

if __name__ == '__main__':
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    setup_seed(2022)
    trainer = Trainer()
    trainer.run()
