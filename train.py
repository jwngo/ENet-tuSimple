import argparse
import json
import os
import shutil
import time
import torch
import torchvision.transforms as transforms 
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

import torch.optim as optim
from torch.utils import data
from tqdm import tqdm
from PIL import Image

import yaml
from dataset.tusimple import tuSimple 
from models.enet import ENet

class Trainer(object): 
    def __init__(self): 

        with open("/Users/jiawei/ENet-tuSimple/config/tusimple_config.yaml") as file: 
            cfg = yaml.load(file, Loader=yaml.FullLoader)
        self.device = torch.device(cfg['DEVICE'])

        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg['DATASET']['MEAN'], cfg['DATASET']['STD']),
        ])
        data_kwargs = {
            'transform': input_transform, 
            'size': cfg['DATASET']['SIZE'],
        } 
        self.train_dataset = tuSimple(
                path=cfg['DATASET']['PATH'],
                image_set='train',
                transforms=input_transform
                ) 
        self.val_dataset = tuSimple(
                path = cfg['DATASET']['PATH'],
                image_set = 'val',
                transforms = input_transform,
                )
        self.iters_per_epoch = len(self.train_dataset) // cfg['TRAIN']['BATCH_SIZE']
        self.max_iters = cfg['TRAIN']['MAX_EPOCHS'] * self.iters_per_epoch

        self.train_sampler = data.sampler.RandomSampler(self.train_dataset) 
        self.val_sampler = data.sampler.RandomSampler(self.val_dataset) 
        
        self.train_batch_sampler = data.sampler.BatchSampler(
            self.train_sampler, 
            cfg['TRAIN']['BATCH_SIZE'],
            drop_last=True,
         )
        self.val_batch_sampler = data.sampler.BatchSampler(
            self.val_sampler,
            cfg['TRAIN']['BATCH_SIZE'],
            drop_last=False
        )

        self.train_loader = data.DataLoader(
            dataset=self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=0,
            pin_memory=True
        )
        self.val_loader = data.DataLoader(
            dataset=self.val_dataset,
            batch_sampler=self.val_batch_sampler,
            num_workers=0,
            pin_memory=True
        )

        # -------- network --------
        self.model = ENet(num_classes=5).to(self.device) 
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg['OPTIM']['LR'],
            weight_decay=cfg['OPTIM']['DECAY'],
        )

    
        

        

                                          
if __name__ == '__main__':
    t = Trainer() 


