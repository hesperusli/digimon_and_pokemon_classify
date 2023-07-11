# 导入 streamlit 包
import streamlit as st
# 导入其他需要的包
import pickle
import PIL.Image
import torch
import Models
import torchvision
from torchvision import transforms
from torch.nn.functional import *
# 定义一个函数，用于加载模型
def load_model():
    # 从 model.pkl 文件中加载模型
    with open("checkpoints/AlexNet.pth", "rb") as f:
        model=Models.getNet()
        checkpoint=torch.load(f)
        model.load_state_dict(checkpoint["state_dict"])
        # model = pickle.load(f)
        # print(model)
    return model
# 创建一个标题
st.title("图片分类器")
# 创建一个副标题
st.subheader("使用训练好的模型对图片进行分类")
# 创建一个文件上传器，用于选择图片文件
uploaded_file = st.file_uploader("请选择一张图片", type=["jpg", "png"])
#定义图片预处理
transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
# 如果选择了图片文件
if uploaded_file is not None:
    # 显示图片
    image = PIL.Image.open(uploaded_file)
    st.image(image, caption="上传的图片", use_column_width=True)
    # st.write(image)
    # 加载模型
    model = load_model()
    model.eval()
    #对图片进行处理
    img=transform(image)
    img=img.unsqueeze(0) # 增加一个批次维度
    # st.write(model)
    # st.write(img.Shape())
    output=model(img)
    prob=softmax(output,dim=1)

    pred=torch.argmax(prob,dim=1)
    # 对图片进行预测
    # prediction = model.predict(image)
    # if pred.item() == 0:
    #     prediction={"daisy"}
    #     # print("The predicted class is: daisy")
    # elif pred.item() == 1:
    #     prediction = {"dandelion"}
    #     # print("The predicted class is: dandelion")
    # elif pred.item() == 2:
    #     prediction = {"roses"}
    #     # print("The predicted class is: roses")
    # elif pred.item() == 3:
    #     prediction = {"sunflowers"}
    #     # print("The predicted class is: sunflowers")
    # elif pred.item() == 4:
    #     prediction = {"tulips"}
        # print("The predicted class is: tulips")
    if pred.item()==0:
        prediction={"digimon"}
    elif pred.item()==1:
        prediction={"pokemon"}
    # 显示预测结果
    st.write(f"预测结果是：{prediction}")