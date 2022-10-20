import pathlib
ROOT = pathlib.Path(__file__).parent.parent

import torch
import numpy as np
import os
from Utils.common_config import Config
import pickle
from tqdm import tqdm

class Dataset():
    def __init__(self , config : Config) -> None:
        if os.path.exists(config.cachepath):
            with open(config.cachepath , 'rb') as f:
                self.train , self.val , self.test = pickle.load(f)
        else:
            self.train = self.parse_file(config.datapath +'_train' , config)
            self.val = self.parse_file(config.datapath +'_val' , config)
            self.test = self.parse_file(config.datapath +'_test' , config)

            with open(config.cachepath , 'wb') as f:
                obj = pickle.dumps([self.train , self.val , self.test])
                f.write(obj)
        
        res = []
        for record in self.train:
            news = {}
            for k , v in record.items():
                news[k] = torch.from_numpy(v)
            res.append(news)
        self.train = res

        res = []
        for record in self.val:
            news = {}
            for k , v in record.items():
                news[k] = torch.from_numpy(v)
            res.append(news)
        self.val = res

        res = []
        for record in self.test:
            news = {}
            for k , v in record.items():
                news[k] = torch.from_numpy(v)
            res.append(news)
        self.test = res
    
    def parse_file(self,filename,config):
        import tensorflow as tf

        dataset = tf.data.TextLineDataset(filename)
        
        def decoding(record , feature_name , feature_default):
            data = tf.io.decode_csv(record , feature_default)
            feature = dict( zip(feature_name , data) )
            label = feature.pop('label')
            return feature , label
        
        dataset = dataset.map(lambda line : decoding(line , config.feature_stastic.keys() , config.feature_default) , num_parallel_calls = 10).batch(config.batch_size)
        
        Data = []
        for data in tqdm(dataset.as_numpy_iterator()):
            record = data[0]
            record['label'] = data[1].astype(np.float32)
            Data.append(record)
        return Data