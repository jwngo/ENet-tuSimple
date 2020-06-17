import argparse
import json
import os
import shutil
import time 
import torch
import torchvision.transforms as transforms 
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.utils import data
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from utils.lane_eval.tusimple_eval import LaneEval
from utils.transforms import * 

import yaml
from dataset.tusimple import tuSimple 
from models.enet import ENet

best_val_loss = 1e6
def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--resume", "-r", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args() 
    return args
args = parse_args() 


class Trainer(object): 
    def __init__(self): 
        cfg_path = os.path.join(os.getcwd(), 'config/tusimple_config.yaml') 
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
        self.train_transform = Compose(Resize(size=(645,373)), RandomCrop(size=(640,368)), Rotation(2), ToTensor(), Normalize(mean=mean, std=std))

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

        # -------- network --------
        weight = [0.4, 1, 1, 1, 1]
        tensor = torch.ones((5,), dtype=torch.float32)
        self.weights = tensor.new_tensor(weight)
        self.model = ENet(num_classes=5).to(self.device) 
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=cfg['OPTIM']['LR'],
            weight_decay=cfg['OPTIM']['DECAY'],
            momentum=0.9,
        )
        #self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1200, gamma=0.8,)
        #self.optimizer = optim.Adam(
        #    self.model.parameters(),
        #    lr = cfg['OPTIM']['LR'],
        #    weight_decay=0,
        #    )
        self.criterion = nn.CrossEntropyLoss(weight=self.weights).cuda() 
        self.bce = nn.BCELoss().cuda()
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

            sem_pred, ins_pred = self.model(img) 
            crossentropy = self.criterion(ins_pred, segLabel) 
            bce = self.bce(sem_pred, segLabel) # TODO BINARY LABEL
            loss = self.criterion(outputs, segLabel)

            self.optimizer.zero_grad() 
            loss.backward() 
            self.optimizer.step()
            #self.lr_scheduler.step()

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
                    "best_val_loss": best_val_loss,
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
        print("Validation loss: {}".format(val_loss)) 
        print("+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*")
        if val_loss < best_val_loss: 
            best_val_loss = val_loss
            save_name = os.path.join(os.getcwd(), 'results', 'run.pth') 
            copy_name = os.path.join(os.getcwd(), 'results', 'run_best.pth') 
            print("val loss is lower than best val loss! Model saved to {}".format(copy_name))
            shutil.copyfile(save_name, copy_name) 
    
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
            for batch_idx, sample in enumerate(test_loader): 
                img = sample['img'].to(self.device) 
                img_name = sample['img_name']
                #segLabel = sample['segLabel'].to(self.device) 
                outputs = self.model(img) 
                count = 0

                # Visualisation 
                for img_idx, _img in enumerate(outputs): 
                    vis = torch.argmax(_img.squeeze(), dim=0).detach().cpu().numpy() 
                    label_colors = np.array([(0,0,0), (255,255,255), (255,128,0), (255,255,0), (128,255,0)])
                    r = np.zeros_like(vis).astype(np.uint8) 
                    g = np.zeros_like(vis).astype(np.uint8) 
                    b = np.zeros_like(vis).astype(np.uint8) 
                    for l in range(0,5):
                        idx = vis == l
                        r[idx] = label_colors[l, 0]
                        g[idx] = label_colors[l, 1]
                        b[idx] = label_colors[l, 2]
                    rgb = np.stack([r,g,b], axis=2) 
                    savename = "{}/{}_{}_vis.png".format(os.path.join(os.getcwd(), 'vis'), batch_idx, count) 
                    count += 1
                    raw_file_name = img_name[img_idx]
                    raw_img = img[img_idx].cpu().detach().numpy()
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
                    
                    #plt.imsave(savename, stack) 
                    # Generate pred.json TODO refactor into another file in  future
                    pred_json = {} 
                    pred_json['lanes'] = []
                    pred_json['h_samples'] = []
                    # truncate everything before 'clips' to be consistent with test_label.json gt
                    pred_json['raw_file'] = raw_file_name[raw_file_name.find('clips'):]
                    pred_json['run_time'] = 0
                    h_samples = [x for x in range(80, 360, 5)]
                    h_sample_actual = [x for x in range(160, 720,10)]
                    # only predicting 4 lanes
                    for i in range(1,5): 
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
                

                                          
if __name__ == '__main__':
    t = Trainer() 

    start_epoch = 0 
    if args.eval == False:
        for epoch in range(start_epoch, t.max_epochs):
            t.train(epoch) 
            if epoch % 1 == 0: 
                print("Validation") 
                t.val(epoch) 
    elif args.eval: 
        # write validation and visualisation here
        save_name = os.path.join(os.getcwd(), 'results', 'run_best.pth')
        save_dict = torch.load(save_name, map_location='cpu') 
        print("Loading", save_name, "from Epoch {}:".format(save_dict['epoch']))
        t.model.load_state_dict(save_dict['model'])
        t.model = t.model.to(t.device)             
        t.eval() 


    


