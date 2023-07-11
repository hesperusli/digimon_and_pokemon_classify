import torch
from torch import nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1=nn.Sequential(  ## 3,256,256
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )                           # 16 123 123
        self.layer2=nn.Sequential(
            torch.nn.Conv2d(16,16,5,2,2),
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)
        )                          # 16 62 62
        self.fc1=nn.Sequential(
            # nn.Linear(16*62*62,120),
            nn.Linear(16*16*16,120),
            nn.Sigmoid()
        )
        self.fc2=nn.Sequential(
            nn.Linear(120,84),
            nn.Sigmoid()
        )
        self.fc3=nn.Sequential(
            nn.Linear(84,40),
            nn.Sigmoid(),
            nn.Linear(84, 2)
        )
        # 前向传播
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x=x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# AlexNet
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=11,
                stride=4,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(32,32,3,padding=1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.fc1=nn.Sequential(
            nn.Linear(1152,120),
            nn.ReLU(),
            nn.Linear(120,5)
        )
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        return x
# VGG
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
# 传入需要的网络列表
def make_features(cfg: list):
    # 创建一个空列表，由于遍历是有顺序的，因此网络结构会按照顺序传入
    layers = []
    # 定义输入图片的通道数
    in_channels = 3
    # 开始遍历列表
    for v in cfg:
        # 如果列表值为M，说明是池化层，就将一个池化层加到空列表中
        if v == "M":
            # 池化核为2×2，步长为2
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # 如果不为M，说明是卷积层
        else:
            # 定义一个卷积层，卷积核个数为列表中的值即为v，卷积核大小为3×3，padding为1
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            # 卷积层紧跟着一个激活函数，因此把卷积层和激活函数都加到空列表中
            layers += [conv2d, nn.ReLU(True)]
            # 上一层的输入为下一层的输出，因此要把in_channels值更新为v
            in_channels = v
    # 将列表通过非关键字参数（把列表中的元素拆开传入）输入到nn.Sequential中生成一个网络结构（也可以通过有序字典的形式输入）
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        # 展平要展1维，因为第0维是batch
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # 遍历网络的每一个子模块
        for m in self.modules():
            # 如果当前层是卷积层
            if isinstance(m, nn.Conv2d):
                # 使用xavier方法初始化卷积核权重
                nn.init.xavier_uniform_(m.weight)
                # 如果卷积核采用了偏置
                if m.bias is not None:
                    # 则将偏置初始化为0
                    nn.init.constant_(m.bias, 0)
            # 若为全连接层
            elif isinstance(m, nn.Linear):
                # 使用xavier方法初始化全连接层权重
                nn.init.xavier_uniform_(m.weight)
                # 将偏置初始化为0
                nn.init.constant_(m.bias, 0)
# 传入需要实例化的网络配置
def vgg(model_name="vgg16", **kwargs):
    # 获得该网络在字典中对应的配置
    cfg = cfgs[model_name]
    # 将配置传入make_features函数得到一个网络，后面的可变长度字典中的参数包括分类个数和是否要初始化网络参数的布尔值
    model = VGG(make_features(cfg), **kwargs)
    return model
# 定义成函数
def getNet():
    model=AlexNet()
    return model