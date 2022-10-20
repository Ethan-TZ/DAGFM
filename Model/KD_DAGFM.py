import torch
from Model import *
from copy import deepcopy

class KD_DAGFM(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.student_network = DAGFM(config)
        self.teacher_network = eval(f"{config.teacher}")(self.get_teacher_config(config))
        self.embedding_layer = self.student_network.embedding_layer = self.teacher_network.embedding_layer
        self.phase = config.phase
        self.alpha = config.alpha
        self.beta = config.beta

        if self.phase != 'teacher_training':
            if not hasattr(config, 'warm_up'):
                raise ValueError("Must have warm up")
            else:
                save_info = torch.load(config.warm_up)
                self.load_state_dict(save_info['model'])


    def get_teacher_config(self, config):
        teacher_cfg = deepcopy(config)
        for property , value in config.__dict__.items():
            if property.startswith('t_'):
                setattr(teacher_cfg,property[2:],value)
        return teacher_cfg

    def FeatureInteraction(self , feature , sparse_input):
        if self.phase == 'teacher_training':
            return self.teacher_network.FeatureInteraction(feature, sparse_input)
        elif self.phase == 'distillation' or self.phase == 'finetuning':
            return self.student_network.FeatureInteraction(feature, sparse_input)
        else:
            raise ValueError("Phase invalid!")

    def forward(self , sparse_input, dense_input = None):
        dense_input = self.embedding_layer(sparse_input)
        if self.phase == 'teacher_training' or self.phase == 'finetuning':
            return self.FeatureInteraction(dense_input, sparse_input)
        elif self.phase == 'distillation':
            if self.training:
                self.t_pred = self.teacher_network(sparse_input)
            return self.FeatureInteraction(dense_input.data, sparse_input)
        else:
            raise ValueError("Phase invalid!")
    
    def calc_loss(self, fetch_data):
        if self.phase == 'teacher_training' or self.phase == 'finetuning':
            prediction = self.forward(fetch_data)
            loss = self.loss_fn(prediction.squeeze(-1) , fetch_data['label'].squeeze(-1).to(self.device))
        elif self.phase == 'distillation':
            self.teacher_network.eval()
            s_pred = self.forward(fetch_data)
            ctr_loss = self.loss_fn(s_pred.squeeze(-1) , fetch_data['label'].squeeze(-1).to(self.device))
            kd_loss = torch.mean( ( self.teacher_network.logits.data - self.student_network.logits ) ** 2 )
            loss =  self.alpha * ctr_loss + self.beta * kd_loss
        else:
            raise ValueError("Phase invalid!")
        return loss