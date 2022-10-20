import torch
from torch import Tensor, nn
from Utils import Config

class DNN(nn.Module):
    def __init__(self , config:Config , Shape , drop_last = True , act = None , autoMatch = True):
        super().__init__()
        layers = []
        self.autoMatch = autoMatch
        if self.autoMatch:
            Shape[0] = (len(config.feature_stastic) -1 ) * config.embedding_size
        for i in range(0 , len(Shape) - 2):
            hidden = nn.Linear(Shape[i] , Shape[i + 1] , bias= True)
            nn.init.normal_(hidden.weight , mean = 0 , std = 0.01)
            layers.append(hidden)
            layers.append(nn.Dropout(p = 0.1))
            layers.append(act if act is not None else nn.ReLU(inplace=False))
        Final = nn.Linear(Shape[-2] , Shape[-1] , bias=True)
        nn.init.xavier_normal_(Final.weight,gain=1.414)
        layers.append(Final)
        layers.append(nn.Dropout(p=0.1))
        if not drop_last:
            layers.append(nn.ReLU(inplace=False))
        self.net = nn.Sequential(*layers)
    
    def forward(self , x : Tensor):
        if self.autoMatch:
            x = x.reshape(x.shape[0] , -1)
        return self.net(x)

