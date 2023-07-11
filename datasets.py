import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from PIL import Image
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
# 将花图片以3：1的方法分配到train和test文件下
data_dir='data'
classes=['digimon','pokemon']
train_dir=os.path.join(data_dir,'train')
test_dir=os.path.join(data_dir,'test')
#
# for cls in classes:
#     os.makedirs(os.path.join(train_dir,cls),exist_ok=True)
#     os.makedirs(os.path.join(test_dir,cls),exist_ok=True)
#
# for cls in classes:
#     src_dir=os.path.join(data_dir,cls)
#     files=os.listdir(src_dir)
#     np.random.shuffle(files)
#     train_files=files[:int(len(files)*0.75)]
#     test_files=files[int(len(files)*0.75):]
#
#     for file in train_files:
#         src_path=os.path.join(src_dir,file)
#         dst_path=os.path.join(train_dir,cls,file)
#         print(src_path)
#         img=Image.open(src_path)
#         img.save(dst_path)
#     for file in test_files:
#         src_path=os.path.join(src_dir,file)
#         dst_path=os.path.join(test_dir,cls,file)
#         print(src_path)
#         img=Image.open(src_path)
#         img.save(dst_path)
# 将图片数据用transforms转化为张量数据
transform=transforms.Compose([
    transforms.Resize((224,224)), #裁剪大小
    # transforms.Resize((256,256)), #裁剪大小
    transforms.ToTensor(),           #转化为tensor变量
    transforms.Normalize((0.5),(0.5))
])
# train_data=torchvision.datasets.ImageFolder('data/automlpoke/train',transform=transform)
# test_data=torchvision.datasets.ImageFolder('data/automlpoke/test',transform=transform)
train_data=torchvision.datasets.ImageFolder('data/train',transform=transform)
test_data=torchvision.datasets.ImageFolder('data/test',transform=transform)
# print(train_data.targets)
# print(train_data.class_to_idx)
print(train_data.class_to_idx) # 输出数据和索引标签