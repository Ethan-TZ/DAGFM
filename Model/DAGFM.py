from os import stat
from turtle import forward
import torch
from torch import embedding, nn
from Model.BasicModel import BasicModel
from Utils import Config
from Model.Layers.DNN import DNN
import numpy as np

class DAGFM(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        type = config.type
        self.depth = config.depth
        field_num = len(config.feature_stastic) - 1
        embedding_size = config.embedding_size
        if type == 'inner':
            self.p = nn.ParameterList([nn.Parameter(torch.randn(field_num , field_num , embedding_size )) for _ in range(self.depth)])
        elif type == 'outer':
            self.p = nn.ParameterList([nn.Parameter(torch.randn(field_num , field_num , embedding_size )) for _ in range(self.depth)])
            self.q = nn.ParameterList([nn.Parameter(torch.randn(field_num , field_num , embedding_size )) for _ in range(self.depth)])
            for _ in range(self.depth):
                nn.init.xavier_normal_(self.p[_] , gain=1.414)
                nn.init.xavier_normal_(self.q[_] , gain=1.414)
        self.adj_matrix = torch.zeros(field_num , field_num, embedding_size).to(self.device)
        for i in range(field_num):
            for j in range(i , field_num):
                self.adj_matrix[i,j,:] += 1
        self.type = type
        self.connect_layer = nn.Parameter(torch.eye(field_num).float())
        self.linear = nn.Linear(field_num * (self.depth + 1) , 1)
        

    def FeatureInteraction(self , feature , sparse_input):
        init_state = self.connect_layer @feature
        h0, ht =  init_state, init_state
        state = [torch.sum(init_state , dim = -1)]
        for i in range(self.depth):
            if self.type == 'inner':
                aggr = torch.einsum('bfd,fsd->bsd', ht , self.p[i] * self.adj_matrix)
                ht = h0 * aggr
            elif self.type == 'outer':
                term = torch.einsum('bfd,fsd->bfs', ht , self.p[i] * self.adj_matrix)
                aggr = torch.einsum('bfs,fsd->bsd', term , self.q[i])
                ht = h0 * aggr
            state.append(torch.sum(ht , -1))
            
        state = torch.cat(state , dim = -1)
        self.logits = self.linear(state)
        self.output = torch.sigmoid(self.logits)
        return self.output

class DeepDAGFM(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        type = config.type
        self.depth = config.depth
        field_num = len(config.feature_stastic) - 1
        embedding_size = config.embedding_size
        if type == 'inner':
            self.p = nn.ParameterList([nn.Parameter(torch.randn(field_num , field_num , embedding_size )) for _ in range(self.depth)])
            for _ in range(self.depth):
                nn.init.xavier_normal_(self.p[_] , gain=1.414)
        elif type == 'outer':
            self.p = nn.ParameterList([nn.Parameter(torch.randn(field_num , field_num , embedding_size )) for _ in range(self.depth)])
            self.q = nn.ParameterList([nn.Parameter(torch.randn(field_num , field_num , embedding_size )) for _ in range(self.depth)])
            for _ in range(self.depth):
                nn.init.xavier_normal_(self.p[_] , gain=1.414)
                nn.init.xavier_normal_(self.q[_] , gain=1.414)
        self.adj_matrix = torch.zeros(field_num , field_num, embedding_size).cuda()
        for i in range(field_num):
            for j in range(i , field_num):
                self.adj_matrix[i,j,:] += 1
        self.type = type
        self.mlplist =  config.mlp
        self.connect_layer = nn.Parameter(torch.eye(field_num).float())
        self.fuse = nn.Linear(embedding_size * field_num , 1)
        self.dnn = DNN(config , self.mlplist)

    def FeatureInteraction(self , feature , sparse_input):
        init_state =  self.connect_layer @ feature
        ht , h0 = init_state , init_state
        state = init_state
        for i in range(self.depth):
            if self.type == 'inner':
                aggr = torch.einsum('bfd,fsd->bsd', ht , self.p[i] * self.adj_matrix)
                ht = h0 * aggr
            elif self.type == 'outer':
                term = torch.einsum('bfd,fsd->bfs', ht , self.p[i] * self.adj_matrix)
                aggr = torch.einsum('bfs,fsd->bsd', term , self.q[i])
                ht = h0 * aggr
            state = state + ht
            
        state = state.view(state.shape[0], -1)

        self.logits = self.dnn(state)
        self.output = torch.sigmoid(self.logits)
        return self.output
    