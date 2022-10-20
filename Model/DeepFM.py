import torch
from torch import nn
from Model.BasicModel import BasicModel
from Utils import Config
from Model.Layers.DNN import DNN
from Model.Layers.LR import LR
import numpy as np

class FM(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.one_order = LR(config)
    
    def FeatureInteraction(self, dense_input, sparse_input):
        fm = torch.sum(dense_input , dim=1)**2 - torch.sum(dense_input ** 2 , dim=1)
        self.logits = torch.sum(0.5 * fm , dim=1 , keepdim=True) + self.one_order(sparse_input)
        self.output = torch.sigmoid(self.logits)
        return self.output

class FwFM(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.one_order = LR(config)
        res = np.arange(len(config.feature_stastic) - 1)
        self.index = []
        for i in range(len(config.feature_stastic) - 2):
            self.index.extend((res[i + 1:] + i * ( len(config.feature_stastic) - 1 ) ).tolist())
        self.field_weight = nn.Parameter(torch.zeros(len(self.index)))
        nn.init.normal_(self.field_weight , mean = 0 , std = 0.01)

    def FeatureInteraction(self, dense_input, sparse_input):
        fwfm = (dense_input[:,:,None,:] * dense_input[:,None,:,:]).reshape(
            dense_input.shape[0] , dense_input.shape[1] ** 2 , dense_input.shape[2]
        )[:,self.index,:] * self.field_weight[None,:,None]
        self.logits = torch.sum(fwfm , dim=(1,2))[:,None] + self.one_order(sparse_input)
        self.output = torch.sigmoid(self.logits)
        return self.output

class FmFM(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        #self.one_order = LR(config)
        dims =( len(config.feature_stastic) - 1 ) * config.embedding_size
        self.mask = torch.zeros(dims , dims).cuda()
        for i in range(0 , len(config.feature_stastic) - 2 ):
            for j in range(i + 1 ,len(config.feature_stastic) - 1 ):
                self.mask[i*config.embedding_size:i * config.embedding_size+config.embedding_size , 
                j*config.embedding_size:j*config.embedding_size+config.embedding_size] += 1
        self.field_weight = nn.Parameter(torch.zeros(dims , dims))
        nn.init.normal_(self.field_weight , mean = 0 , std = 0.01)
    
    def FeatureInteraction(self, dense_input, sparse_input):
        dense_input = dense_input.reshape(dense_input.shape[0],dense_input.shape[1] * dense_input.shape[2])
        fmfm = dense_input @ (self.field_weight * self.mask) * dense_input
        self.logits = torch.sum(fmfm ,dim=1 , keepdim=True) + self.one_order(sparse_input)
        self.output = torch.sigmoid(self.logits)
        return self.output

class DeepFM(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.one_order = LR(config)
        self.dnn = DNN(config , config.mlp)
    
    def FeatureInteraction(self, dense_input, sparse_input, *kwargs):
        fm = torch.sum(dense_input , dim=1)**2 - torch.sum(dense_input ** 2 , dim=1)
        self.logits = torch.sum(0.5 * fm , dim=1 , keepdim=True) + self.one_order(sparse_input) \
            + self.dnn(dense_input)
        self.output = torch.sigmoid(self.logits)
        return self.output
