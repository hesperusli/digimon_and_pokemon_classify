import torch
from torchvision import transforms
import torchvision
import Models
import config
from PIL import Image
from torch.nn.functional import *


# classes ={0:"daisy",1:"dandelion",2:"roses",3:"sunflowers",4:"tulips"}
classes ={0:"digimon",1:"pokemon"}

# 加载模型
# model=torch.load(config.weights+config.model+'.pth')
model=Models.getNet()
checkpoint=torch.load("checkpoints/AlexNet.pth")
model.load_state_dict(checkpoint["state_dict"])
# print(model)
model.eval()
# 定义图片预处理
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
])
# 选择一张图片
# img=Image.open("example/1.jpg")
# 改进方法，可以输入一个图片地址，进行地址选择，然后分类
img_path = input("Please enter the image path: ")  # 使用input进行输入
img=Image.open(img_path)
# img=torchvision.io.read_image(image_path)
print(img)
# 对图片进行预处理
img=transform(img)
img=img.unsqueeze(0) # 增加一个批次维度
# 将图片输入模型，得到输出
output=model(img)
#对输出进行softmax，得到概率分布
prob=softmax(output,dim=1)

# 找打最大概率的索引，预测的类别
pred=torch.argmax(prob,dim=1)
print("The predicted class is:", pred.item())
# 打印预测结果
# if pred.item() == 0:
#     print("The predicted class is: daisy")
# elif pred.item() == 1:
#     print("The predicted class is: dandelion")
# elif pred.item() == 2:
#     print("The predicted class is: roses")
# elif pred.item() == 3:
#     print("The predicted class is: sunflowers")
# elif pred.item() == 4:
#     print("The predicted class is: tulips")
if pred.item() ==0:
    print("The predicted class is: digimon")
elif pred.item()==1:
    print("The predicted class is: pokemon")

# print("The predicted class is:", pred.item())
