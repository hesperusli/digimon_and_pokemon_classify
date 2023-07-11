import torch
import numpy as np
import torchvision
import os
from torch.utils.data import DataLoader
from torch import nn,optim
import config
import Models
import datasets
from utils import utils
from torch.autograd import Variable
from test import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ =='__main__':
    # 检查文件夹是否存在，没有就创建
    if not os.path.exists(config.example):
        os.mkdir(config.example)
    if not os.path.exists(config.weights):
        os.mkdir(config.weight)
    if not os.path.exists(config.logs):
        os.mkdir(config.weight)
    # 定义模型
    model=Models.getNet()
    if torch.cuda.is_available():
        model=model.cuda()
    # 定义优化器
    optimizer=optim.SGD(model.parameters(),lr=config.lr,weight_decay=config.lr_decay,momentum=0.9)
    # 定义损失函数  二元交叉熵损失
    # loss_BCE=nn.BCELoss().to(device)
    # loss_BCE=nn.CrossEntropyLoss().to(device)
    loss_criterion = nn.CrossEntropyLoss().to(device)
    # 使用BCEWithLogitsLoss()
    # loss_BCEWithLogitsLoss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0])).to(device)
    # 检查是否需要加载checkpoint已经训练好的模型
    start_epoch=0
    current_accuracy=0
    resume=False    # 默认不加载
    if resume:
        checkpoint=torch.load(config.weights+config.model+'.pth')
        start_epoch=checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    # 加载训练数据集和测试数据集
    train_loader=DataLoader(datasets.train_data,batch_size=config.bach_size,shuffle=True,num_workers=0)
    test_loader=DataLoader(datasets.test_data,batch_size=config.bach_size,shuffle=True,num_workers=0)
    # 开始训练
    #初始化loss和acc
    train_loss=[]
    acc=[]
    test_loss=[]
    #创建一个列表来保存训练训练损失的值
    train_loss_list=[]
    #创建一个列表来保存训练准确度的值
    train_acc_list=[]
    print("-------------------------------start Training--------------------------------------")
    for epoch in range(start_epoch,config.epochs):
        model.train()
        lr=utils.lr_step(epoch)
        optimizer=optim.SGD(model.parameters(),lr=lr,weight_decay=config.weight_decay,momentum=0.9)
        loss_epoch=0
        for index,(input,target) in enumerate(train_loader):
            model.train()
            input=Variable(input).to(device)
            # m=nn.Sigmoid()
            target=Variable(torch.from_numpy(np.array(target))).long().to(device)
            #将目标转化为one_hot编码的形式
            # target_onehot=F.one_hot(target,num_classes=2)
            # 梯度清零
            optimizer.zero_grad()
            output=model(input)
            # loss=loss_BCE(m(output),target)
            loss=loss_criterion(output,target)
            #反向传播和梯度更新
            loss.backward()
            optimizer.step()

            loss_epoch+=loss.item()*input.size(0)
            if (index+1)%10 == 0:
                # print("Epoch:{}[{:>3d}/{}]\t Loss:{:6f} ".format(epoch+1,index*config.bach_size,
                #                                                 len(train_loader.dataset),loss.item()))
                print("Epoch: {} [{:>3d}/{}]\t Loss: {:.6f} ".format(epoch + 1, index * config.bach_size,
                                                                     len(train_loader.dataset), loss.item()))
        loss_epoch = loss_epoch/len(train_loader.dataset)
        print("loss:",loss_epoch)
        train_loss_list.append(loss_epoch)
        if (epoch + 1) % 1 == 0:
            print("--------------------------------------Eavluate-------------------------------")
            model.eval()
            # 测试使用测试数据集
            test_loss1, accTop1 = evaluate(test_loader, model, loss_criterion)
            acc.append(accTop1)
            # train_loss_list.append(accTop1)
            print("type(accTop1) =", type(accTop1))
            print(accTop1)
            test_loss.append(test_loss1)
            train_loss.append(loss_epoch / len(train_loader))
            print("Test_epoch: {} Test_accurary: {:.4} Test_loss: {:.6f}".format(epoch + 1, accTop1, test_loss1))
            save_model = accTop1 > current_accuracy
            accTop1 = max(current_accuracy, accTop1)
            current_accuracy = accTop1
            utils.save_checkpoint({
                "epoch": epoch + 1,
                "model_name": config.model,
                "state_dict": model.state_dict(),
                "accTop1": current_accuracy,
                "optimizer": optimizer.state_dict(),
            }, save_model)
            # 绘制损失和准确度曲线
            fig = plt.figure(figsize=(10, 5))
            # 在图形上添加一个子图，用于绘制损失曲线
            ax1 = fig.add_subplot(1, 2, 1)  # 1行2列的第1个子图

            # 绘制损失曲线，设置颜色，标签和线宽等参数
            ax1.plot(train_loss_list, color="blue", label="Train Loss", linewidth=2)

            # 设置子图的标题，x轴标签和y轴标签等参数
            ax1.set_title("Loss Curve")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")

            # 在图形上添加一个子图，用于绘制准确率曲线
            ax2 = fig.add_subplot(1, 2, 2)  # 1行2列的第2个子图

            # 绘制准确率曲线，设置颜色，标签和线宽等参数
            ax2.plot(acc, color="red", label="Train Accuracy", linewidth=2)

            # 设置子图的标题，x轴标签和y轴标签等参数
            ax2.set_title("Accuracy Curve")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")

            # 显示图形
            plt.show()



