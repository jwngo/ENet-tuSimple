import argparse
import json
import os
import shutil
import time 
import torch
import torchvision.transforms as transforms 
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import datetime
import sys

import torch.optim as optim
from torch.utils import data
from tqdm import tqdm
from PIL import Image
from utils.lane_eval.tusimple_eval import LaneEval
from utils.lane_eval import getLane
from utils.lane_eval.intersection import SegmentationMetric
from utils.transforms import * 
from utils.scheduler.lr_scheduler import get_scheduler
from torch.utils.tensorboard import SummaryWriter

import yaml
from dataset.tusimple import tuSimple 
from models.enet import ENet
best_val_loss = 1e6
best_mIoU = 0
original_stdout = sys.stdout
def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--resume", "-r", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--val", action="store_true") 
    parser.add_argument("exp_name", help="name of experiment")
    args = parser.parse_args() 
    return args
args = parse_args() 

class Trainer(object): 
    def __init__(self, exp): 
        # IoU and pixAcc Metric calculator
        self.metric = SegmentationMetric(7)
        cfg_path = os.path.join(os.getcwd(), 'config/tusimple_config.yaml') 
        self.exp_name = exp
        self.writer = SummaryWriter('tensorboard/' + self.exp_name)
        with open(cfg_path) as file: 
            cfg = yaml.load(file, Loader=yaml.FullLoader)
        self.device = torch.device(cfg['DEVICE'])
        self.max_epochs = cfg['TRAIN']['MAX_EPOCHS']
        self.dataset_path = cfg['DATASET']['PATH']
        # TODO remove this and refactor PROPERLY
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg['DATASET']['MEAN'], cfg['DATASET']['STD']),
        ])

        mean = cfg['DATASET']['MEAN']
        std = cfg['DATASET']['STD']
        self.train_transform = Compose(Resize(size=(645,373)), RandomCrop(size=(640,368)), RandomFlip(0.5), Rotation(2), ToTensor(), Normalize(mean=mean, std=std))

        self.val_transform = Compose(Resize(size=(640,368)), ToTensor(), Normalize(mean=mean, std=std))
        data_kwargs = {
            'transform': self.input_transform, 
            'size': cfg['DATASET']['SIZE'],
        } 
        self.train_dataset = tuSimple(
                path=cfg['DATASET']['PATH'],
                image_set='train',
                transforms=self.train_transform
                ) 
        self.val_dataset = tuSimple(
                path = cfg['DATASET']['PATH'],
                image_set = 'val',
                transforms =self.val_transform,
                )
        self.train_loader = data.DataLoader(
                dataset = self.train_dataset,
                batch_size = cfg['TRAIN']['BATCH_SIZE'],
                shuffle = True,
                num_workers = 0,
                pin_memory = True,
                drop_last = True,
                )
        self.val_loader = data.DataLoader(
                dataset = self.val_dataset,
                batch_size = cfg['TRAIN']['BATCH_SIZE'],
                shuffle = False,
                num_workers = 0, 
                pin_memory = True,
                drop_last = False,
                ) 
        self.iters_per_epoch = len(self.train_dataset) // (cfg['TRAIN']['BATCH_SIZE'])
        self.max_iters = cfg['TRAIN']['MAX_EPOCHS'] * self.iters_per_epoch
        # -------- network --------
        weight = [0.4, 1, 1, 1, 1, 1, 1]
        self.model = ENet(num_classes=7).to(self.device) 
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=cfg['OPTIM']['LR'],
            weight_decay=cfg['OPTIM']['DECAY'],
            momentum=0.9,
        )
        self.lr_scheduler = get_scheduler(self.optimizer, max_iters=self.max_iters, iters_per_epoch=self.iters_per_epoch)
        #self.optimizer = optim.Adam(
        #    self.model.parameters(),
        #    lr = cfg['OPTIM']['LR'],
        #    weight_decay=0,
        #    )
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4, 1, 1, 1, 1, 1, 1])).cuda() 
        self.bce = nn.BCELoss().cuda()
    def train(self, epoch, start_time):
        running_loss = 0.0
        is_better = True
        prev_loss = float('inf') 
        logging.info('Start training, Total Epochs: {:d}, Total Iterations: {:d}'.format(self.max_epochs, self.max_iters))
        print("Train Epoch: {}".format(epoch))
        self.model.train() 
        epoch_loss = 0
        #progressbar = tqdm(range(len(self.train_loader)))
        iteration = epoch * self.iters_per_epoch if epoch > 0 else 0
        start_time = start_time
        for batch_idx, sample in enumerate(self.train_loader): 
            iteration += 1
            img = sample['img'].to(self.device) 
            segLabel = sample['segLabel'].to(self.device) 
            exist = sample['exist'].to(self.device)
            # outputs is crossentropy, sig is binary cross entropy
            outputs, sig = self.model(img) 
            ce = self.criterion(outputs,segLabel)
            bce = self.bce(sig, exist)
            loss = ce + (0.1 * bce) 


            self.optimizer.zero_grad() 
            loss.backward() 
            self.optimizer.step()
            self.lr_scheduler.step()
            #print("LR", self.optimizer.param_groups[0]['lr'])

            epoch_loss += loss.item() 
            running_loss += loss.item() 
            eta_seconds = ((time.time() - start_time) / iteration) * (self.max_iters - iteration) 
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            iter_idx = epoch * len(self.train_loader) + batch_idx
            #progressbar.set_description("Batch loss: {:.3f}".format(loss.item()))
            #progressbar.update(1)
            # Tensorboard
            if iteration % 10 == 0:
                logging.info(
                "Epoch: {:d}/{:d} || Iters: {:d}/{:d} || Lr: {:6f} || "
                "Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                epoch, self.max_epochs, iteration % self.iters_per_epoch, self.iters_per_epoch, 
                self.optimizer.param_groups[0]['lr'], loss.item(), str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))
            if batch_idx % 10 == 9: 
                self.writer.add_scalar('train loss',
                                running_loss / 10,
                                epoch * len(self.train_loader) + batch_idx + 1)
                running_loss = 0.0
        #progressbar.close() 
        if epoch % 1 == 0: 
            save_dict = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optim": self.optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    }
            save_name = os.path.join(os.getcwd(), 'results', self.exp_name, 'run.pth')
            save_name_epoch = os.path.join(os.getcwd(), 'results', self.exp_name, '{}.pth'.format(epoch))
            torch.save(save_dict, save_name) 
            torch.save(save_dict, save_name_epoch) 
            print("Model is saved: {}".format(save_name))
            print("Model is saved: {}".format(save_name_epoch))
            print("+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*")
        return epoch_loss/len(self.train_loader)
    def val(self, epoch, train_loss):
        self.metric.reset()
        global best_val_loss
        global best_mIoU
        print("Val Epoch: {}".format(epoch))
        self.model.eval()
        val_loss = 0 
        #progressbar = tqdm(range(len(self.val_loader)))
        with torch.no_grad(): 
            for batch_idx, sample in enumerate(self.val_loader):
                img = sample['img'].to(self.device) 
                segLabel = sample['segLabel'].to(self.device) 
                exist = sample['exist'].to(self.device)
                outputs, sig = self.model(img) 
                ce = self.criterion(outputs, segLabel)
                bce = self.bce(sig, exist)
                loss = ce + (0.1*bce) 
                val_loss += loss.item() 
                self.metric.update(outputs, segLabel)
                pixAcc, mIoU = self.metric.get()
                logging.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                    batch_idx + 1, pixAcc * 100, mIoU * 100))
                #progressbar.set_description("Batch loss: {:3f}".format(loss.item()))
                #progressbar.update(1)
                # Tensorboard
                #if batch_idx + 1 == len(self.val_loader):
                #    self.writer.add_scalar('train - val loss',
                #                    train_loss - (val_loss / len(self.val_loader)),
                #                    epoch)
        #progressbar.close() 
        pixAcc, mIoU, category_iou = self.metric.get(return_category_iou = True)
        print(category_iou)
        logging.info('End validation pixAcc: {:.3f}, mIoU: {:.3f}'.format(
            pixAcc * 100, mIoU * 100))
        iter_idx = (epoch + 1) * len(self.train_loader)
        with open('val_out.txt', 'a') as out:
            sys.stdout = out
            print(self.exp_name, 'Epoch:', epoch, 'pixAcc: {:.3f}, mIoU: {:.3f}'.format(pixAcc*100, mIoU*100))
            sys.stdout = original_stdout
        print("Validation loss: {}".format(val_loss)) 
        print("+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*")
        if (mIoU * 100) > best_mIoU:
            best_mIoU = mIoU*100
            save_dict = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optim": self.optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_mIoU": best_mIoU,
                    }
            save_name = os.path.join(os.getcwd(), 'results', self.exp_name, 'best_mIoU.pth')
            torch.save(save_dict, save_name)
            print("mIoU is higher than best mIoU! Model saved to {}".format(save_name))
        #if val_loss < best_val_loss: 
        #    best_val_loss = val_loss
        #    save_name = os.path.join(os.getcwd(), 'results', self.exp_name, 'run.pth') 
        #    copy_name = os.path.join(os.getcwd(), 'results', self.exp_name, 'run_best.pth') 
        #    print("val loss is lower than best val loss! Model saved to {}".format(copy_name))
        #    shutil.copyfile(save_name, copy_name) 
    
    def eval(self):
        print("+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*")
        print("Evaluating.. ") 
        self.model.eval() 
        val_loss = 0 
        dump_to_json = [] 
        test_dataset = tuSimple(
                path=self.dataset_path,
                image_set='test',
                transforms=self.val_transform
                ) 
        test_loader = data.DataLoader(
                dataset = test_dataset,
                batch_size = 12, 
                shuffle = False,
                num_workers = 0, 
                pin_memory = True,
                drop_last = False,
                ) 
        progressbar = tqdm(range(len(test_loader))) 
        with torch.no_grad():
            with open('exist_out.txt','w') as f:
                for batch_idx, sample in enumerate(test_loader): 
                    img = sample['img'].to(self.device) 
                    img_name = sample['img_name']
                    #segLabel = sample['segLabel'].to(self.device) 
                    outputs, sig = self.model(img) 
                    seg_pred = F.softmax(outputs, dim=1)
                    seg_pred = seg_pred.detach().cpu().numpy()
                    exist_pred = sig.detach().cpu().numpy()
                    count = 0

                    for img_idx in range(len(seg_pred)):
                        seg = seg_pred[img_idx]
                        exist = [1 if exist_pred[img_idx ,i] > 0.5 else 0 for i in range(6)]
                        lane_coords = getLane.prob2lines_tusimple(seg, exist, resize_shape=(720,1280), y_px_gap=10, pts=56)
                        for i in range(len(lane_coords)):
                            # sort lane coords
                            lane_coords[i] = sorted(lane_coords[i], key=lambda pair:pair[1])
                        
                        #print(len(lane_coords))
                    # Visualisation 
                        savename = "{}/{}_{}_vis.png".format(os.path.join(os.getcwd(), 'vis'), batch_idx, count) 
                        count += 1
                        raw_file_name = img_name[img_idx]
                        pred_json = {}
                        pred_json['lanes'] = []
                        pred_json['h_samples'] = []
                        # truncate everything before 'clips' to be consistent with test_label.json gt
                        pred_json['raw_file'] = raw_file_name[raw_file_name.find('clips'):]
                        pred_json['run_time'] = 0

                        for l in lane_coords:
                            empty = all(lane[0] == -2 for lane in l)
                            if len(l)==0:
                                continue
                            if empty:
                                continue
                            pred_json['lanes'].append([])
                            for (x,y) in l:
                                pred_json['lanes'][-1].append(int(x))
                        for (x, y) in lane_coords[0]:
                            pred_json['h_samples'].append(int(y))
                        dump_to_json.append(json.dumps(pred_json))
                    progressbar.update(1)
                progressbar.close() 

                with open(os.path.join(os.getcwd(), "results", self.exp_name, "pred_json.json"), "w") as f:
                    for line in dump_to_json:
                        print(line, end="\n", file=f)

                print("Saved pred_json.json to {}".format(os.path.join(os.getcwd(), 'results', self.exp_name, "pred_json.json")))
           
                '''
                        raw_img = img[b].cpu().detach().numpy()
                        raw_img = raw_img.transpose(1, 2, 0)
                        # Normalize both to 0..1
                        min_val, max_val = np.min(raw_img), np.max(raw_img)
                        raw_img = (raw_img - min_val) / (max_val - min_val)
                        #rgb = rgb / 255.
                        #stack = np.hstack((raw_img, rgb))
                        background = Image.fromarray(np.uint8(raw_img*255))
                        overlay = Image.fromarray(rgb)
                        new_img = Image.blend(background, overlay, 0.4)
                        new_img.save(savename, "PNG")
                '''
                        
                '''
                        # Generate pred.json TODO refactor into another file in  future
                        pred_json = {} 
                        pred_json['lanes'] = []
                        pred_json['h_samples'] = []
                        # truncate everything before 'clips' to be consistent with test_label.json gt
                        pred_json['raw_file'] = raw_file_name[raw_file_name.find('clips'):]
                        pred_json['run_time'] = 0
                        h_samples = [x for x in range(80, 360, 5)]
                        h_sample_actual = [x for x in range(160, 720,10)]
                        # predicting 6 lanes
                        for i in range(1,7): 
                            pred_json['lanes'].append([])
                            ii = np.nonzero(vis == i)
                            x, y = ii[1], ii[0]
                            coordinates = dict()   
                            # can use collections here to make more 'pythonic' TODO
                            for x, y in zip(x,y): 
                                if y in h_samples:
                                # multiply by 2 since our resolution is 640x368, gt is 1280x720
                                    if y*2 in coordinates:
                                        coordinates[y*2].append(int(x*2))
                                    else: 
                                        coordinates[y*2] = [x*2]

                            for y_actual in h_sample_actual: 
                                if y_actual not in coordinates:
                                    pred_json['lanes'][-1].append(-2)
                                else: 
                                    # Take the middle of all pixels in this y_coordinate
                                    pred_json['lanes'][-1].append(int(coordinates[y_actual][len(coordinates[y_actual])//2]))

                            empty = all(lane == -2 for lane in pred_json['lanes'][-1])
                            if empty:
                                pred_json['lanes'].pop(-1)
                                continue
                        pred_json['h_samples'] = h_sample_actual
                        #print(pred_json) 
                        dump_to_json.append(json.dumps(pred_json))
                    #loss = self.criterion(outputs, segLabel) 
                    #val_loss += loss.item() 
                    #progressbar.set_description("Batch loss: {:3f}".format(loss.item()))
                    progressbar.update(1)
            progressbar.close() 
            with open(os.path.join(os.getcwd(), "pred_json.json"), "w") as f:
                for line in dump_to_json:
                    print(line, end="\n", file=f)

            print("Saved pred_json.json to {}".format(os.path.join(os.getcwd(), "pred_json.json")))
            print("Validation loss: {}".format(val_loss))
            print("+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*")
            print("Evaluating with TuSimple benchmark eval..") 
            eval_result = LaneEval.bench_one_submit(os.path.join(os.getcwd(), "pred_json.json"), "/mnt/4TB/ngoj0003/ENet-tuSimple/data_tusimple/dataset/test_label.json")
            print(eval_result)
            with open(os.path.join(os.getcwd(), "evaluation_result.txt"), "w") as f: 
                print(eval_result, file=f)
        '''
                
if __name__ == '__main__':
    t = Trainer(args.exp_name) 
    os.makedirs(os.path.join(os.getcwd(), 'results', t.exp_name), exist_ok=True)
    logging.basicConfig(filename=str(os.path.join(os.getcwd(), 'results', t.exp_name) + '/log.txt'), level=logging.INFO) 
    console = logging.StreamHandler() 
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(__name__)

    start_epoch = 0 
    start_time = time.time() 
    if args.val: 
        for f in os.listdir(os.path.join(os.getcwd(), 'results', t.exp_name)):
            if f.endswith(".pth") and len(f) == 7:
                save_name = os.path.join(os.getcwd(), 'results', t.exp_name, f)
                save_dict = torch.load(save_name, map_location='cpu')
                print("Loading", save_name, "from Epoch {}:".format(save_dict['epoch']))
                t.model.load_state_dict(save_dict['model'])
                epoch = save_dict['epoch']
                epoch_train_loss = 0
                t.model = t.model.to(t.device)             
                t.val(epoch, epoch_train_loss) 

    elif args.eval == False:
        if args.resume:
            save_dict = torch.load(os.path.join(os.getcwd(), 'results', t.exp_name, 'run.pth'))
            print("Loaded {}!".format(os.path.join(os.getcwd(),'results', t.exp_name, 'run.pth')))
            t.model.load_state_dict(save_dict['model'])
            t.optimizer.load_state_dict(save_dict['optim'])
            start_epoch = save_dict['epoch']
            best_val_loss = save_dict['best_val_loss']

        else:
            epoch = 0
        for epoch in range(start_epoch, t.max_epochs):
            epoch_train_loss = t.train(epoch, start_time) 
            if epoch % 1 == 0: 
                print("Validation") 
                t.val(epoch, epoch_train_loss) 
    elif args.eval: 
        save_name = os.path.join(os.getcwd(), 'results', t.exp_name, 'best_mIoU.pth')
        save_dict = torch.load(save_name, map_location='cpu') 
        print("Loading", save_name, "from Epoch {}:".format(save_dict['epoch']))
        t.model.load_state_dict(save_dict['model'])
        t.model = t.model.to(t.device)             
        t.eval() 
