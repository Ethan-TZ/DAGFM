import torch
from torch import nn
from Model.BasicModel import BasicModel
from Utils import Config
from Model.Layers.DNN import DNN


class CrossNetComp(nn.Module):
    def __init__(self, config:Config):
        super(CrossNetComp, self).__init__()
        hiddenSize = (len(config.feature_stastic) - 1) * config.embedding_size
        self.W = nn.Linear(hiddenSize , hiddenSize)
        nn.init.xavier_normal_(self.W.weight , gain=1.414)
        
    def forward(self, base, cross):
        result = base * self.W(cross) + cross
        return result

class CrossNet(BasicModel):
    def __init__(self , config: Config) -> None:
        super().__init__(config)
        self.depth = config.depth
        self.crossnet = nn.ModuleList([CrossNetComp(config) for i in range(self.depth)])
        self.linear = nn.Linear((len(config.feature_stastic) - 1) * config.embedding_size , 1)
        nn.init.normal_(self.linear.weight)
        self.backbone = ['crossnet','linear', 'embedding_layer']
    
    def FeatureInteraction(self , feature , sparse_input):
        feature = feature.reshape(feature.shape[0] , -1)
        base = feature
        cross = feature
        for i in range(self.depth):
            cross = self.crossnet[i](base , cross)
        self.logits = self.linear(cross)
        self.output = torch.sigmoid(self.logits)
        return self.output

class DCNV2(BasicModel):
    def __init__(self , config: Config) -> None:
        super().__init__(config)
        self.depth = config.depth
        self.mlplist = config.mlp
        self.dnn = DNN(config , self.mlplist)
        self.crossnet = nn.ModuleList([CrossNetComp(config) for i in range(self.depth)])
        self.linear = nn.Linear((len(config.feature_stastic) - 1) * config.embedding_size , 1)
        self.backbone = ['dnn','crossnet','linear']
    
    def FeatureInteraction(self , feature , sparse_input, *kargs):
        dnn = self.dnn(feature)
        feature = feature.reshape(feature.shape[0] , -1)
        base = feature
        cross = feature
        for i in range(self.depth):
            cross = self.crossnet[i](base , cross)
        
        self.logits = self.linear(cross) + dnn
        self.output = torch.sigmoid(self.logits)
        return self.output
