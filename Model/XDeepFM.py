import torch
from torch import nn
from Model.BasicModel import BasicModel
from Utils import Config
from Model.Layers.DNN import DNN
from Model.Layers.LR import LR

class CINComp(nn.Module):
    def __init__(self, indim , outdim , config: Config):
        super(CINComp, self).__init__()
        basedim = len(config.feature_stastic) - 1
        self.conv = nn.Conv1d(indim * basedim , outdim,1)
    def forward(self, infeature, base):
        return self.conv(
            (infeature[:,:,None,:] * base[:,None,:,:]) \
            .reshape(infeature.shape[0] , infeature.shape[1] * base.shape[1],-1)
        )

class XDeepFM(BasicModel):
    def __init__(self , config: Config) -> None:
        super().__init__(config)
        self.mlplist = config.mlp
        self.dnn = DNN(config , self.mlplist)
        self.one_order = LR(config)
        self.cinlist = [len(config.feature_stastic) - 1] + config.cin
        self.cin = nn.ModuleList([CINComp(self.cinlist[i] , self.cinlist[i + 1],config) for i in range(0 , len(self.cinlist) - 1)])

        self.linear = nn.Parameter( torch.zeros(sum(self.cinlist) - self.cinlist[0] , 1 ) )
        nn.init.normal_(self.linear, mean=0, std=0.01)
        self.backbone = ['dnn' , 'one_order' , 'cin' , 'linear']

    def FeatureInteraction(self , feature , sparse_input, * kargs):
        dnn = self.dnn(feature)
        base = feature
        x = feature
        p = []
        for comp in self.cin:
            x = comp(x , base)
            p.append(torch.sum(x , dim=-1))
        p = torch.cat(p , dim=-1)
        cin = p @ self.linear
        self.logits = cin + dnn
        self.output = torch.sigmoid(self.logits)
        return self.output

class CIN(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.cinlist = [len(config.feature_stastic) - 1] + config.cin
        self.cin = nn.ModuleList([CINComp(self.cinlist[i] , self.cinlist[i + 1],config) for i in range(0 , len(self.cinlist) - 1)])
        self.linear = nn.Parameter( torch.zeros(sum(self.cinlist) - self.cinlist[0] , 1 ) )
        nn.init.normal_(self.linear, mean=0, std=0.01)
        self.backbone = ['cin' , 'linear']
    
    def FeatureInteraction(self , feature , sparse_input, *kargs):
        base = feature
        x = feature
        p = []
        for comp in self.cin:
            x = comp(x , base)
            p.append(torch.sum(x , dim=-1))
        p = torch.cat(p , dim=-1)
        self.logits = p @ self.linear
        self.output = torch.sigmoid(self.logits)
        return self.output
