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
        cfg_path = os.path.join(os.getcwd(), 'config/tusimple_config.yaml') 
        with open(cfg_path) as file: 
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
                
        self.train_loader = data.DataLoader(
                dataset = self.train_dataset,
                batch_size = 16,
                shuffle = True,
                num_workers = 0,
                pin_memory = True,
                drop_last = True,
                )
        self.val_loader = data.DataLoader(
                dataset = self.val_dataset,
                batch_size = 16, 
                shuffle = True,
                num_workers = 0, 
                pin_memory = True,
                drop_last = False,
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


