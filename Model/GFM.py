import torch
from torch import nn
from Model.BasicModel import BasicModel

import torch.nn.functional as F

class GFM(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.depth = len(config.block_shape)
        self.ks = config.topks
        self.embedding_dim = config.embedding_size
        self.num_fields = len(config.feature_stastic) - 1
        self.head_num = config.head_num
        self.block_shape = config.block_shape
        self.select_mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(config.embedding_size if i == 0 else self.block_shape[i-1] , 16),
            nn.ReLU(),
            nn.Linear(16 , 1),
            nn.Sigmoid())
            for i in range(self.depth)])
        self.attention_mlp = nn.ModuleList([
            nn.Linear(config.embedding_size if i == 0 else self.block_shape[i-1], self.head_num)
            for i in range(self.depth)])

        self.W = nn.ModuleList([
            nn.Linear(config.embedding_size if i == 0 else self.block_shape[i-1], self.block_shape[i])
            for i in range(self.depth)])
        
        self.R = nn.ParameterList([
            nn.Parameter(torch.randn(self.num_fields,self.block_shape[i],config.embedding_size if i == 0 else self.block_shape[i-1]))
            for i in range(self.depth)])        
        self.b = nn.ParameterList([
            nn.Parameter(torch.randn(1,self.num_fields,self.block_shape[i]))
            for i in range(self.depth)])      
        self.outlayer = nn.Linear(sum(self.block_shape) , 1)

        rows , cols = [] , [] 
        for i in range(self.num_fields):
            for j in range(self.num_fields):
                rows.append(i)
                cols.append(j)
        self.rows , self.cols = rows , cols

    def FeatureInteraction(self , feature , sparse_input):
        h = feature
        state = []
        for i in range(self.depth):
            M = h[:,self.rows,:] * h[:,self.cols,:] #[b,e,d]
            Q = (self.R[i] @ h.unsqueeze(-1)).squeeze(-1) + self.b[i]

            Q = torch.relu(Q)
            S = self.select_mlp[i](M).reshape(-1 , self.num_fields , self.num_fields) #[b,f,f]
            values , indices = S.topk(self.ks[i] , dim = -1) #[b , f , k]
            kth , _ = torch.min(values , dim = -1 , keepdim= True) #[b ,f , 1]
            mask = (S >= kth).long() #[b,f,f]
            S = S * mask
            #mask = torch.tile(1-mask, dims=(self.head_num,1,1)).unsqueeze(-1).bool()
            S = torch.tile(S, dims=(self.head_num,1,1)).unsqueeze(-1) #[bh,f,f,1]

            A = self.attention_mlp[i](M).reshape(-1 , self.num_fields , self.num_fields, self.head_num) #[b,f,f,h]
            A = torch.cat(torch.split(A, [1] * self.head_num , dim=-1), dim=0) #[bh,f,f,1]

            H = self.W[i](M).reshape(-1 , self.num_fields , self.num_fields, self.block_shape[i]) #[b,f,f,D]
            H = torch.cat(torch.split(H, [self.block_shape[i] // self.head_num] * self.head_num , dim=-1), dim=0) #[bh,f,f,D/h]
            
            A = F.softmax(A * S,dim=-2) #[bh,f,f,1]
            h = torch.sum(H * A , dim=2) #[bh,f,D/h]
            h = torch.cat( torch.split(h , [h.shape[0] // self.head_num] * self.head_num , dim=0) , dim=-1 ) #[b,f,D]
            h = torch.relu(h + Q)
            h = torch.layer_norm(h,[self.block_shape[i]],eps=1e-8)
            state.append(h)
        y = torch.cat(state , dim=-1)
        y = torch.mean(y , dim = 1) #[b,D]
        self.logits = self.outlayer(y) #[b,1]
        self.output = torch.sigmoid(self.logits)
        return self.output