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
from dataset import get_segmentation_dataset
from models import ENet

class Trainer(object): 
    def __init__(self, args): 

        with open("ENet-tuSimple/config/tusimple_config.yaml") as file: 
            cfg = yaml.load(file, Loader=yaml.FullLoader)
        self.args = args
        self.device = torch.device(args.device)

        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg['DATASET']['MEAN'], cfg['DATASET']['STD']),
        ])
        data_kwargs = {
            'transform': input_transform, 
            'size': cfg['SIZE'],
        } 
        train_dataset = get_segmentation_dataset(
            'tusimple',
            path=cfg['DATASET']['PATH'],
            image_set='train',
            transforms=input_transform,
        )
        val_dataset = get_segmentation_dataset(
            'tusimple',
            path=cfg['DATASET']['PATH'],
            image_set='val',
            transforms=input_transform,
        )
        self.iters_per_epoch = len(train_dataset) // cfg['TRAIN']['BATCH_SIZE']
        self.max_iters = cfg['TRAIN']['MAX_EPOCHS'] * self.iters_per_epoch

        self.train_sampler = data.sampler.RandomSampler(train_dataset) 
        self.val_sampler = data.sampler.RandomSampler(val_dataset) 
        
        self.train_batch_sampler = data.sampler.IterationBasedBatchSampler(
            train_sampler, 
            cfg['TRAIN']['BATCH_SIZE'],
            self.max_iters,
            drop_last=True,
         )
        self.val_batch_sampler = data.sampler.BatchSampler(
            val_sampler,
            cfg['TRAIN']['BATCH_SIZE'],
            drop_last=False
        )

        self.train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=0,
            pin_memory=True
        )
        self.val_loader = data.DataLoader(
            dataset=val_dataset,
            batch_sampler=val_batch_sampler,
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

    
        

        

def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./runs")
    parser.add_argument("--resume", "-r", type=bool, default=False) 
    parser.add_argument("--device", "-d", default="cuda:0")
    args = parser.parse_args() 
    return args
args = parse_args() 

# -------- config --------
out_dir = args.out_dir
device = torch.device(cfg['DEVICE'])

# -------- train data --------
mean = cfg['DATASET']['MEAN']
std = cfg['DATASET']['STD']
height = cfg['SIZE'][0]
width = cfg['SIZE'][1]
                                          
if __name__ == '__main__':
    print('ok')


