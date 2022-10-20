from abc import abstractmethod
from audioop import bias
from turtle import forward
import torch
from torch import nn
from Model.BasicModel import BasicModel
from Utils import Config
from Model.Layers.DNN import DNN
import numpy as np
from .AutoInt import AttentionLayer
from itertools import product
import torch.nn.functional as F

class FiGNN(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.embedding_dim = config.embedding_size
        self.num_fields = len(config.feature_stastic) - 1
        self.fignn = FiGNN_Layer(self.num_fields, 
                                 self.embedding_dim,
                                 gnn_layers=config.depth,
                                 reuse_graph_layer=True,
                                 use_gru=True,
                                 use_residual=True,
                                 device=torch.device('cpu'))#torch.device('cuda:0'))
        self.fc = AttentionalPrediction(self.num_fields, self.embedding_dim)
        self.output_activation = nn.Sigmoid()
                    
    def FeatureInteraction(self , feature , sparse_input):
        feature_emb = feature
        h_out = self.fignn(feature_emb)
        self.logits = self.fc(h_out)
        self.output = self.output_activation(self.logits)
        return self.output


class AttentionalPrediction(nn.Module):
    def __init__(self, num_fields, embedding_dim):
        super(AttentionalPrediction, self).__init__()
        self.mlp1 = nn.Linear(embedding_dim, 1, bias=False)
        self.mlp2 = nn.Sequential(nn.Linear(num_fields * embedding_dim, num_fields, bias=False),
                                  nn.Sigmoid())

    def forward(self, h):
        score = self.mlp1(h).squeeze(-1) # b x f
        weight = self.mlp2(h.flatten(start_dim=1)) # b x f
        logit = (weight * score).sum(dim=1).unsqueeze(-1)
        return logit

class FiGNN_Layer(nn.Module):
    def __init__(self, 
                 num_fields, 
                 embedding_dim,
                 gnn_layers=3,
                 reuse_graph_layer=False,
                 use_gru=True,
                 use_residual=True,
                 device=None):
        super(FiGNN_Layer, self).__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.gnn_layers = gnn_layers
        self.use_residual = use_residual
        self.reuse_graph_layer = reuse_graph_layer
        self.device = device
        if reuse_graph_layer:
            self.gnn = GraphLayer(num_fields, embedding_dim)
        else:
            self.gnn = nn.ModuleList([GraphLayer(num_fields, embedding_dim)
                                      for _ in range(gnn_layers)])
        self.gru = nn.GRUCell(embedding_dim, embedding_dim) if use_gru else None
        self.src_nodes, self.dst_nodes = zip(*list(product(range(num_fields), repeat=2)))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.W_attn = nn.Linear(embedding_dim * 2, 1, bias=False)

    def build_graph_with_attention(self, feature_emb):
        src_emb = feature_emb[:, self.src_nodes, :]
        dst_emb = feature_emb[:, self.dst_nodes, :]
        concat_emb = torch.cat([src_emb, dst_emb], dim=-1)
        alpha = self.leaky_relu(self.W_attn(concat_emb))
        alpha = alpha.view(-1, self.num_fields, self.num_fields)
        mask = torch.eye(self.num_fields).to(self.device)
        alpha = alpha.masked_fill(mask.bool(), float('-inf'))
        graph = F.softmax(alpha, dim=-1) # batch x field x field without self-loops
        return graph

    def forward(self, feature_emb):
        g = self.build_graph_with_attention(feature_emb)
        h = feature_emb
        for i in range(self.gnn_layers):
            if self.reuse_graph_layer:
                a = self.gnn(g, h)
            else:
                a = self.gnn[i](g, h)
            if self.gru is not None:
                a = a.view(-1, self.embedding_dim)
                h = h.view(-1, self.embedding_dim)
                h = self.gru(a, h)
                h = h.view(-1, self.num_fields, self.embedding_dim)
            else:
                h = a + h
            if self.use_residual:
                h += feature_emb
        return h


class GraphLayer(nn.Module):
    def __init__(self, num_fields, embedding_dim):
        super(GraphLayer, self).__init__()
        self.W_in = torch.nn.Parameter(torch.Tensor(num_fields, embedding_dim, embedding_dim))
        self.W_out = torch.nn.Parameter(torch.Tensor(num_fields, embedding_dim, embedding_dim))
        nn.init.xavier_normal_(self.W_in)
        nn.init.xavier_normal_(self.W_out)
        self.bias_p = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, g, h):
        h_out = torch.matmul(self.W_out, h.unsqueeze(-1)).squeeze(-1) # broadcast multiply
        aggr = torch.bmm(g, h_out)
        a = torch.matmul(self.W_in, aggr.unsqueeze(-1)).squeeze(-1) + self.bias_p
        return a