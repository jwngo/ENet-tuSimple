import argparse 
import json 
import os
import shutil
import time
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler 
import torch.nn.functional as F
import numpy as np
import logging
import time
import datetime
import sys
import torch.optim as optim
import yaml

from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.lane_eval.tusimple_eval import LaneEval
from utils.lane_eval import getLane
from utils.lane_eval.intersection import SegmentationMetric
from utils.transforms import *
from utils.scheduler.lr_scheduler import get_scheduler
from dataset.tusimple import tuSimple
from models.enet import ENet

best_mIoU = 0
best_val_loss = 1e6
original_stdout = sys.stdout
def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--resume", "-r", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("s_exp_name", help="name of student exp")
    parser.add_argument("t_exp_name", help="name of teacher exp")
    args = parser.parse_args()
    return args
args = parse_args() 

class Trainer(object): 
    def __init__(self, s_exp_name, t_exp_name):
        cfg_path = os.path.join(os.getcwd(), 'config/tusimple_config.yaml')
        self.s_exp_name = s_exp_name
        self.t_exp_name = t_exp_name
        self.writer = SummaryWriter('tensorboard/' + self.s_exp_name)
        self.metric = SegmentationMetric(7)
        with open(cfg_path) as cfg: 
            config = yaml.load(cfg, Loader=yaml.FullLoader)
        self.device = torch.device(config['DEVICE'])
        self.max_epochs = config['TRAIN']['MAX_EPOCHS'] 
        self.dataset_path = config['DATASET']['PATH']
        self.mean = config['DATASET']['MEAN']
        self.std = config['DATASET']['STD']
        '''
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
            ])
        '''
        self.train_transform = Compose(
            Resize(size=(645,373)),
            RandomCrop(size=(640,368)),
            RandomFlip(0.5),
            Rotation(2),
            ToTensor(),
            Normalize(mean=self.mean,std=self.std))
        self.val_transform = Compose(
            Resize(size=(640,368)),
            ToTensor(),
            Normalize(mean=self.mean,std=self.std)
            )
        self.train_dataset = tuSimple(
            path=config['DATASET']['PATH'],
            image_set='train',
            transforms=self.train_transform
            )
        self.val_dataset = tuSimple(
            path=config['DATASET']['PATH'],
            image_set='val',
            transforms=self.val_transform,
            )
        self.train_loader = data.DataLoader(
            dataset=self.train_dataset,
            batch_size=config['TRAIN']['BATCH_SIZE'],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            )
        self.val_loader = data.DataLoader(
            dataset=self.val_dataset,
            batch_size=config['TRAIN']['BATCH_SIZE'],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            )
        self.iters_per_epoch = len(self.train_dataset) // config['TRAIN']['BATCH_SIZE']
        self.max_iters = self.max_epochs * self.iters_per_epoch

        # ------------network------------
        self.s_model = ENet(num_classes=7).to(self.device)
        self.t_model = ENet(num_classes=7).to(self.device)
        self.optimizer = optim.SGD(
            self.s_model.parameters(),
            lr=config['OPTIM']['LR'],
            weight_decay=config['OPTIM']['DECAY'],
            momentum=0.9,
            )
        self.lr_scheduler = get_scheduler(
            self.optimizer,
            max_iters=self.max_iters,
            iters_per_epoch=self.iters_per_epoch,
            )
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor([0.4,1,1,1,1,1,1])).cuda() #background weight 0.4
        self.bce = nn.BCELoss().cuda()
        self.kl = nn.KLDivLoss().cuda()#reduction='batchmean' gives NaN
        self.mse = nn.MSELoss().cuda()

    def train(self, epoch, start_time):
        running_loss = 0.0
        is_better = True
        prev_loss = float('inf')
        logging.info('Start training, Total Epochs: {:d}, Total Iterations: {:d}'.format(self.max_epochs, self.max_iters))
        print("Train Epoch: {}".format(epoch))
        self.s_model.train()
        self.t_model.eval()
        epoch_loss = 0
        iteration = epoch*self.iters_per_epoch if epoch>0 else 0
        start_time = start_time
        for batch_idx, sample in enumerate(self.train_loader):
            iteration+=1
            img = sample['img'].to(self.device)
            segLabel = sample['segLabel'].to(self.device)
            exist = sample['exist'].to(self.device)
            with torch.no_grad(): 
                t_outputs, t_sig = self.t_model(img)
            s_outputs, s_sig = self.s_model(img)
            ce = self.ce(s_outputs, segLabel)
            bce = self.bce(s_sig, exist)
            kl = self.kl(
                F.log_softmax(s_outputs, dim=1),
                F.softmax(t_outputs, dim=1),
                ) 
            mse = self.mse(s_outputs, t_outputs) #/ s_outputs.size(0)
            loss = ce + (0.1*bce) + kl + (0.5*mse)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            epoch_loss += loss.item()
            running_loss += loss.item()
            eta_seconds = ((time.time() - start_time)/iteration) * (self.max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            iter_idx = epoch*len(self.train_loader) + batch_idx
            if iteration%10==0:
                logging.info(
                    "Epoch: {:d}/{:d} || Iters: {:d}/{:d} || Lr: {:6f} || Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                    epoch,
                    self.max_epochs,
                    iteration % self.iters_per_epoch,
                    self.iters_per_epoch,
                    self.optimizer.param_groups[0]['lr'],
                    loss.item(),
                    str(datetime.timedelta(seconds=int(time.time() - start_time))),
                    eta_string,
                    ))
            if batch_idx % 10 == 9:
                self.writer.add_scalar(
                    'train_loss',
                    running_loss/10,
                    epoch * len(self.train_loader) + batch_idx + 1)
                running_loss = 0.0
        if epoch % 1 == 0:
            save_dict = {
                "epoch": epoch,
                "model": self.s_model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "best_mIoU": best_mIoU,
                "best_val_loss": best_val_loss,
                }
            save_name = os.path.join(os.getcwd(), 'results', self.s_exp_name, 'run.pth')
            torch.save(save_dict, save_name)
            print("Model is saved: {}".format(save_name))

    def val(self, epoch):
        self.metric.reset() 
        global best_val_loss
        global best_mIoU
        print("Val Epoch: {}".format(epoch))
        self.s_model.eval()
        val_loss = 0 
        with torch.no_grad(): 
            for batch_idx, sample in enumerate(self.val_loader):
                img = sample['img'].to(self.device) 
                segLabel = sample['segLabel'].to(self.device)
                exist = sample['exist'].to(self.device)
                outputs, sig = self.s_model(img)
                ce = self.ce(outputs, segLabel) 
                bce = self.bce(sig, exist) 
                loss = ce + (0.1*bce)
                self.metric.update(outputs, segLabel)
                pixAcc, mIoU = self.metric.get() 
                logging.info("Sample: {:d}, pixAcc: {:.3f}, mIoU: {:.3f}".format(
                batch_idx + 1,
                pixAcc * 100,
                mIoU * 100
                ))

        pixAcc, mIoU, category_iou = self.metric.get(
            return_category_iou=True
            )
        print(category_iou)
        logging.info("Final pixAcc: {:.3f}, mIoU: {:.3f}".format(
            pixAcc * 100,
            mIoU * 100,
            ))
        iter_idx = (epoch+1)*len(self.train_loader)
        if (mIoU*100) > best_mIoU:
            best_mIoU = mIoU*100
            save_dict = {
                "epoch": epoch,
                "model": self.s_model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_mIoU": best_mIoU,
                }
            save_name = os.path.join(os.getcwd(), 'results', self.s_exp_name, 'best_mIoU.pth')
            torch.save(save_dict, save_name)
            print("mIoU is higher than best mIoU! Model saved to {}".format(save_name))




if __name__ == '__main__':
    t = Trainer(args.s_exp_name, args.t_exp_name)
    os.makedirs(os.path.join(os.getcwd(), 'results', t.s_exp_name),exist_ok=True)
    logging.basicConfig(filename=str(os.path.join(os.getcwd(), 'results', t.s_exp_name) + '/log.txt'), level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(__name__)
    start_epoch = 0
    start_time = time.time() 
    epoch = 0
    save_name = os.path.join(os.getcwd(), 'results', t.t_exp_name, '268.pth')
    save_dict = torch.load(save_name, map_location='cpu')
    print("Loading", save_name, "from Epoch {}:".format(save_dict['epoch']))
    t.t_model.load_state_dict(save_dict['model'])
    t.t_model = t.t_model.to(t.device)
    for epoch in range(start_epoch, t.max_epochs):
    # Can remove epoch_train_loss
        t.train(epoch, start_time)
        t.val(epoch)
