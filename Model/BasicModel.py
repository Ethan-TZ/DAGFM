from abc import abstractmethod
from cmath import log
import torch
from torch import embedding, nn
from Model.Layers.Embedding import Embedding
from abc import abstractmethod

class BasicModel(nn.Module):
    def __init__(self , config, device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        super().__init__()
        self.device = device
        self.L2_weight = config.L2
        self.embedding_layer = Embedding(config)
        self.loss_fn = nn.BCELoss()
        self.backbone = []

    def forward(self , sparse_input, dense_input = None):
        dense_input = self.embedding_layer(sparse_input)
        predict = self.FeatureInteraction(dense_input , sparse_input)
        return predict
    
    @abstractmethod
    def FeatureInteraction(self , dense_input , sparse_input, *kwrds):
        pass

    def L2_Loss(self , weight):
        if weight == 0:
            return 0
        loss = 0
        for _ in self.backbone:
            comp = getattr(self , _)
            if isinstance(comp , nn.Parameter):
                loss += torch.norm(comp , p = 2)
                continue
            for params in comp.parameters():
                loss += torch.norm(params , p = 2)
        return loss * weight
    
    def calc_loss(self, fetch_data):
        prediction = self.forward(fetch_data)
        loss = self.loss_fn(prediction.squeeze(-1) , fetch_data['label'].squeeze(-1).to(self.device)) + self.L2_Loss(self.L2_weight)
        return loss