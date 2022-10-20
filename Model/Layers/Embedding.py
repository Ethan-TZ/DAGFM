import torch
from torch import nn
from Utils import Config

class Embedding(nn.Module):
    def __init__(self , config : Config):
        super().__init__()
        self.embedding: nn.ModuleDict[str:nn.Embedding] = nn.ModuleDict()

        # self.user_field = config.user_field
        # self.item_field = config.item_field
        # self.pop = {}

        for feature , numb in config.feature_stastic.items():
            if feature != 'label':
                self.embedding[feature] = nn.Embedding(numb + 1 , config.embedding_size)
                # if feature == self.user_field or feature == self.item_field:
                #     self.pop[feature] = torch.zeros(numb + 1).cuda()
                #     self.embedding[feature+'meta'] = nn.Embedding(1 , config.embedding_size)
                # else:
                #     self.embedding[feature+'meta'] = nn.Embedding(numb + 1 , config.embedding_size)
        for _, value in self.embedding.items():
            nn.init.xavier_uniform_(value.weight)
        
        
    def get_popularity(self , train_data):
        #return
        for batch in train_data:
            for j in batch[self.user_field]:
                self.pop[self.user_field][j] += 1
            for j in batch[self.item_field]:
                self.pop[self.item_field][j] += 1            
        self.pop[self.user_field] /= torch.sum(self.pop[self.user_field])
        self.pop[self.item_field] /= torch.sum(self.pop[self.item_field])

    def forward(self , data, mode = "normal"):
        batch = len(data['label'])
        out = []
        if mode == "normal":
            for name , raw in data.items():
                if name != 'label':
                        out.append(self.embedding[name](raw.long().cuda())[:,None,:] )#+ (self.embedding[name+'meta'].weight[None,:] if name + 'meta' in self.embedding else 0))

        else:
            for name , raw in data.items():
                if name != 'label':
                    if name == self.user_field or name == self.item_field :#or name in self.context_field:
                        #out.append(torch.repeat_interleave(self.embedding[name+'meta'].weight[None,:], batch,0))
                        out.append( torch.repeat_interleave(torch.sum(self.embedding[name].weight * (self.pop[name][:,None]) , dim = 0)[None,None,:], batch,0) )
                    else:
                        out.append(self.embedding[name](raw.long().cuda())[:,None,:])

        return torch.cat(out , dim = -2)