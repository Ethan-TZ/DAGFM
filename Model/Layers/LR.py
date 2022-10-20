import torch
from torch import Tensor, nn
from Utils import Config

class LR(nn.Module):
    def __init__(self , config:Config):
        super(LR,self).__init__()
        self.embedding: nn.ModuleDict[str:nn.Embedding] = nn.ModuleDict()
        for feature , numb in config.feature_stastic.items():
            if feature != 'label':
                self.embedding[feature] = torch.nn.Embedding(numb+1 , 1)
        
        for _, value in self.embedding.items():
            nn.init.xavier_normal_(value.weight)
    
    def forward(self , data):
        out = []
        for name , raw in data.items():
            if name != 'label':
                out.append(self.embedding[name](raw.long().cuda())[:,None,:])
        out = torch.cat(out , dim = -2)
        return torch.sum(out , dim = 1)