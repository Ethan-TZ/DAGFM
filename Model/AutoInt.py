import torch
from torch import nn
from Model.BasicModel import BasicModel
from Utils import Config
from Model.Layers.DNN import DNN

class AttentionLayer(nn.Module):
    def __init__(self , headNum = 2 , att_emb = 16 , input_emb = 16):
        super().__init__()
        self.headNum = headNum
        self.att_emb = att_emb
        self.input_emb = input_emb
        self.Query = nn.Parameter(torch.zeros(1 , self.headNum , 1 , self.input_emb , self.att_emb))
        self.Key = nn.Parameter(torch.zeros(1 , self.headNum , 1 , self.input_emb , self.att_emb))
        self.Value = nn.Parameter(torch.zeros(1 , self.headNum , 1 , self.input_emb , self.att_emb))
        self.res = nn.Parameter(torch.zeros(self.input_emb , self.headNum * self.att_emb))
        self.init()

    def forward(self, feature):
        res = feature @ self.res
        feature = feature.view(feature.shape[0] , 1 , feature.shape[1] , 1 , -1 )
        query = (feature @ self.Query).squeeze(3)
        key = (feature @ self.Key).squeeze(3)
        value = (feature @ self.Value).squeeze(3)

        score = torch.softmax(query @ key.transpose(-1,-2) , dim = -1)
        em = score @ value
        em = torch.transpose(em , 1  , 2)
        em = em.reshape(res.shape[0],res.shape[1],res.shape[2])
        
        return torch.relu(em + res)
    
    def init(self):
        for params in self.parameters():
            nn.init.xavier_uniform_(params , gain=1.414)

class AutoInt(BasicModel):
    def __init__(self, config : Config):
        super().__init__(config)
        self.featureNum = len(config.feature_stastic) - 1
        self.featureDim = config.embedding_size
        self.headNum = config.headNum
        self.LayerNum = config.LayerNum
        self.att_emb = config.att_emb
        self.input_emb = self.att_emb * self.headNum
        self.interacting = nn.Sequential(*[AttentionLayer(self.headNum , self.att_emb , self.input_emb if _ != 0 else self.featureDim) for _ in range(self.LayerNum)])
        self.mlp = config.mlp
        self.dnn = DNN(config , self.mlp)
        self.linear = nn.Linear(self.mlp[-1] + self.input_emb * self.featureNum , 1)
        nn.init.xavier_uniform_(self.linear.weight , gain=1.414)
        self.backbone = ['interacting' , 'attention_embedding' , 'dnn','linear']

    def FeatureInteraction(self , feature , sparse_input, *kwards):
        dnn = self.dnn(feature)        
        attention = self.interacting(feature) #[b,f,h*d]
        attention = attention.view(feature.shape[0] , -1)
        xfinal = torch.cat([dnn , attention] , dim=1)
        self.logits = self.linear(xfinal)
        self.output = torch.sigmoid(self.logits)
        return self.output