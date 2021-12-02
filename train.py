# -*- coding: utf-8 -*-
# Author: Changxing DENG
# @Time: 2021/10/8 11:27

import glob
import os
import random
import time
import xml.dom.minidom as xdm

import numpy as np
import torch
import torch.nn as nn
from sacred import Experiment
from sacred.observers import MongoObserver
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import VOCDataset
from network import SSDNet
from loss import SSDLoss

EXPERIMENT_NAME = "SSD"
ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(MongoObserver(url="11.11.11.100:11220", db_name="sacred"))

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def load_data(data_path):
    voc2007_trainval_annotations = os.path.join(data_path, "VOC2007", "trainval", "Annotations", "*xml")
    voc2012_trainval_annotations = os.path.join(data_path, "VOC2012", "trainval", "Annotations", "*xml")
    annotation_path = glob.glob(voc2007_trainval_annotations) + glob.glob(voc2012_trainval_annotations)

    all_category = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                     "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                     "train", "tvmonitor"]

    random.shuffle(annotation_path)
    res = []
    for i, path in enumerate(annotation_path):
        path_split = os.path.dirname(path).split('/')
        folder = [path_split[2], path_split[3]] # [VOC2007/VOC2012, trainval/test]
        DOMTree = xdm.parse(path)
        collection = DOMTree.documentElement
        filename = collection.getElementsByTagName("filename")[0].childNodes[0].data
        category_list = []
        bndbox_list = []
        object_ = collection.getElementsByTagName("object")
        for obj in object_:
            category = obj.getElementsByTagName("name")[0].childNodes[0].data
            bndbox = obj.getElementsByTagName("bndbox")[0]
            xmin = int(bndbox.getElementsByTagName("xmin")[0].childNodes[0].data)
            ymin = int(bndbox.getElementsByTagName("ymin")[0].childNodes[0].data)
            xmax = int(bndbox.getElementsByTagName("xmax")[0].childNodes[0].data)
            ymax = int(bndbox.getElementsByTagName("ymax")[0].childNodes[0].data)
            category_list.append(all_category.index(category))
            bndbox_list.append([xmin, ymin, xmax, ymax])
        res.append({"folder": folder, "filename": filename, "category": np.array(category_list), "bndbox": np.array(bndbox_list)})
    return res

def collate_fn(batch):
    # To handle the situation when __getitem__ in dataset.py return []
    img_list = []
    target_list = []
    for img, target in batch:
        if img == []:
            continue
        img_list.append(img)
        target_list.append(target)
    return torch.stack(img_list, dim=0), torch.stack(target_list, dim=0)

@ex.automain
def main(_run):
    set_random_seeds(42)
    DATA_PATH = "../dataset"
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-4
    EPOCH = 1000
    PRINT_INTERVAL = 50
    SAVE_ROOT = os.path.join("models", EXPERIMENT_NAME, str(_run._id))
    os.makedirs(SAVE_ROOT, exist_ok=True)
    #-----数据加载-----#
    res = load_data(DATA_PATH)
    split = int(len(res)*0.8)
    train_res = res[:split]
    eval_res = res[split:]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    train_set = VOCDataset(data=train_res, image_root=DATA_PATH, transform=transform, train=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=8, drop_last=True, shuffle=True, collate_fn=collate_fn)
    eval_set = VOCDataset(data=eval_res, image_root=DATA_PATH, transform=transform, train=False)
    eval_loader = DataLoader(eval_set, batch_size=BATCH_SIZE, num_workers=8, drop_last=False, shuffle=False, collate_fn=collate_fn)
    #-----模型加载-----#
    net = SSDNet()
    net = nn.DataParallel(net)
    net = net.cuda()
    #-----优化-----#
    criterion = SSDLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    #-----迭代训练/验证-----#
    step = 0
    best_loss = 1e6
    for epoch in range(EPOCH):
        net.train()
        if epoch == 25:
            optimizer.param_groups[0]['lr'] = 1e-4
        if epoch == 45:
            optimizer.param_groups[0]['lr'] = 1e-5
        for iter_, (img, target) in enumerate(train_loader):
            img = img.cuda()
            target = target.cuda() # bs, 8096, 5
            out = net(img) # bs, 8096, 25
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter_ % PRINT_INTERVAL == 0:
                print("Time: {}, Epoch: [{}/{}], Iter: [{}/{}], Loss: {}"
                      .format(time.strftime("%m-%d %H:%M:%S", time.localtime()), epoch, EPOCH, iter_, len(train_loader), loss.item()), flush=True)
                _run.log_scalar("Train Loss", loss.item(), step=step)
            step += 1
        total_loss = 0
        net.eval()
        with torch.no_grad():
            for _, (img, target) in enumerate(eval_loader):
                img = img.cuda()
                target = target.cuda()
                out = net(img)
                loss = criterion(out, target)
                total_loss += loss.item() * img.shape[0]
        avg_loss = total_loss / len(eval_set)
        print("Time: {}, Epoch: [{}/{}], Loss: {}".format(time.strftime("%m-%d %H:%M:%S", time.localtime()), epoch, EPOCH, avg_loss), flush=True)
        _run.log_scalar("Eval Loss", avg_loss, step=step)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(net.state_dict(), f"{SAVE_ROOT}/{epoch}_{step}.pth")
