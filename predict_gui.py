import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import Models
# 加载预训练的ResNet模型
# resnet = models.resnet18(pretrained=True)
# resnet.eval()
model=Models.getNet()
checkpoint=torch.load("checkpoints/CNN.pth")
model.load_state_dict(checkpoint["state_dict"])
# print(model)
model.eval()

# 定义图片预处理的变化
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.9),(0.9))
])
# 创建界面的标题和说明
st.title("Image Classification Web App using PyTorch and Streamlit")
st.header("Upload an image and the app will predict its class")
st.text("The app uses a pretrained ResNet model from PyTorch")

# 创建一个选择按钮，用于上传图片文件
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png"])

# 如果有上传文件，执行以下操作
if uploaded_file is not None:
    # 打开图片文件并显示图片
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 对图片进行预处理并转换为张量
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    # 对图片进行分类并得到输出张量
    output = model(image_tensor)

    # 根据输出张量，得到分类结果并显示结果
    _, prediction = torch.max(output, 1)
    class_index = prediction.item()
    # class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    class_names={0:"daisy",1:"dandelion",2:"roses",3:"sunflowers",4:"tulips"}
    class_name = class_names[class_index]
    st.write(f"The predicted class is: {class_name}")