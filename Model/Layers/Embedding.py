import torch
from torch import nn
from Utils import Config

class Embedding(nn.Module):
    def __init__(self , config : Config, device):
        super().__init__()
        self.device = device
        self.embedding: nn.ModuleDict[str:nn.Embedding] = nn.ModuleDict()
        for feature , numb in config.feature_stastic.items():
            if feature != 'label':
                self.embedding[feature] = nn.Embedding(numb + 1 , config.embedding_size)
        for _, value in self.embedding.items():
            nn.init.xavier_uniform_(value.weight)

    def forward(self , data, mode = "normal"):
        out = []
        for name , raw in data.items():
            if name != 'label':
                    out.append(self.embedding[name](raw.long().to(self.device))[:,None,:] )
        return torch.cat(out , dim = -2)