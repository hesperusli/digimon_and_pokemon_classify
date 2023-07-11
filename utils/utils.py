import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import config
import datetime
# 设置不同的epoch，不同的学习率
def lr_step(epoch):
    if epoch<30:
        lr=0.01
    if epoch<80:
        lr=0.001
    if epoch<120:
        lr=0.0005
    else:
        lr=0.0001
    return lr

#保存模型
def save_checkpoint(state,save_model):
    filename=config.weights+config.model+".pth"
    torch.save(state,filename)
    if save_model:
        message=config.weights+config.model+".pth"
        print("Get Better top1 : % s saveing weights to %s"%(state["accTop1"],message),datetime.datetime.now())
        with open("./logs/%s.txt"%config.model,"a") as f:
            print("Get Better top1: %s saving weights to %s"%(state["accTop1"],message),datetime.datetime.now(),
                  file=f)
