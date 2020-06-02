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

best_val_loss = 1e6

class Trainer(object): 
    def __init__(self): 
        cfg_path = os.path.join(os.getcwd(), 'config/tusimple_config.yaml') 
        with open(cfg_path) as file: 
            cfg = yaml.load(file, Loader=yaml.FullLoader)
        self.device = torch.device(cfg['DEVICE'])
        self.max_epochs = cfg['TRAIN']['MAX_EPOCHS']

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
                shuffle = False,
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
        self.criterion = nn.CrossEntropyLoss().cuda() 
    def train(self, epoch):
        is_better = True
        prev_loss = float('inf') 
        print("Train Epoch: {}".format(epoch))
        self.model.train() 
        epoch_loss = 0
        progressbar = tqdm(range(len(self.train_loader)))

        for batch_idx, sample in enumerate(self.train_loader): 
            img = sample['img'].to(self.device) 
            segLabel = sample['segLabel'].to(self.device) 

            outputs = self.model(img) 
            loss = self.criterion(outputs, segLabel)

            self.optimizer.zero_grad() 
            loss.backward() 
            self.optimizer.step() 

            epoch_loss += loss.item() 
            iter_idx = epoch * len(self.train_loader) + batch_idx
            progressbar.set_description("Batch loss: {:.3f}".format(loss.item()))
            progressbar.update(1)

        progressbar.close() 
        if epoch % 1 == 0: 
            save_dict = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optim": self.optimizer.state_dict(),
                    "best_val_loss": best_val_loss
                    }
           
            save_name = os.path.join(os.getcwd(), 'results', 'run.pth')
            torch.save(save_dict, save_name) 
            print("Model is saved: {}".format(save_name))
            print("+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*")
    def val(self, epoch):
        global best_val_loss

        print("Val Epoch: {}".format(epoch))

        self.model.eval()
        val_loss = 0 
        progressbar = tqdm(range(len(self.val_loader)))

        with torch.no_grad(): 
            for batch_idx, sample in enumerate(self.val_loader):
                img = sample['img'].to(self.device) 
                segLabel = sample['segLabel'].to(self.device) 
                
                outputs = self.model(img) 
                loss = self.criterion(outputs, segLabel) 
                val_loss += loss.item() 
                progressbar.set_description("Batch loss: {:3f}".format(loss.item()))
                progressbar.update(1)
        progressbar.close() 
        iter_idx = (epoch + 1) * len(self.train_loader) 
        print(val_loss)
        print("+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*")
        if val_loss < best_val_loss: 
            best_val_loss = val_loss
            save_name = os.path.join(os.getcwd(), 'results', 'run.pth') 
            copy_name = os.path.join(os.getcwd(), 'results', 'run_best.pth') 
            print("val loss is lower than best val loss! Model saved to {}".format(copy_name))
            shutil.copyfile(save_name, copy_name) 


    
            

        
    
                                          
if __name__ == '__main__':
    t = Trainer() 

    start_epoch = 0 
    for epoch in range(start_epoch, t.max_epochs):
        t.train(epoch) 
        if epoch % 1 == 0: 
            print("Validation") 
            t.val(epoch) 


    


