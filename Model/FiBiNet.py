import torch
from torch import nn
from Model.BasicModel import BasicModel
from Utils import Config
from Model.Layers.DNN import DNN
import numpy as np

class BiLinearComp(nn.Module):
    def __init__(self , config : Config):
        super().__init__()
        self.kernel_matrix = nn.Parameter(torch.zeros(1 , len(config.feature_stastic) - 1 , config.embedding_size , config.embedding_size))
        nn.init.normal_(self.kernel_matrix , mean = 0 ,std = 0.01)
        res = np.arange(len(config.feature_stastic) - 1)
        self.index = []
        for i in range(len(config.feature_stastic) - 2):
            self.index.extend((res[i + 1:] + i * ( len(config.feature_stastic) - 1 ) ).tolist())
    
    def forward(self , feature):
        return ( (feature[:,:,None,:] @  self.kernel_matrix) * feature[:,None,:,:] ).\
        reshape(feature.shape[0] , feature.shape[1] * feature.shape[1] , feature.shape[2])[:,self.index,:]

class SENetComp(nn.Module):
    def __init__(self, config : Config):
        super().__init__()
        self.dnn = DNN(config , [len(config.feature_stastic) - 1 , (len(config.feature_stastic) - 1) // 8 , len(config.feature_stastic) - 1],
        autoMatch=False)
    
    def forward(self , feature):
        imp = torch.mean(feature , dim = -1)
        imp = self.dnn(imp)
        return feature * torch.sigmoid(imp[:,:,None])

class FiBiNet(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.left_bilin = BiLinearComp(config)
        self.right_bilin = BiLinearComp(config)
        self.senet = SENetComp(config)
        self.mlplist = config.mlp
        self.mlplist[0] = (len(config.feature_stastic) - 1) * (len(config.feature_stastic) - 2) * config.embedding_size 
        self.dnn = DNN(config , Shape=self.mlplist ,autoMatch=False)

        self.backbone = ['dnn','left_bilin','right_bilin','senet']
    
    def FeatureInteraction(self , feature , sparse_input):
        expf = self.senet(feature)
        right = self.right_bilin(expf)
        left = self.left_bilin(feature)
        input = torch.cat([left,right] , dim=1).reshape(feature.shape[0] , -1)
        self.logits = self.dnn(input)
        self.output = torch.sigmoid(self.logits)
        return self.output
