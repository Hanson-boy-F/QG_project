# ResNet神经网络模型
import torch.nn as nn
import torch


# 定义一个类--18或34层神经网络的残差残存结构
class BasicBlock(nn.Module):
    expansion = 1   # 这个参数对应我们主分支采用的卷积层使用的卷积核个数是否变化，18和34层一样，50，101，152层不一样
    # 定义初始函数    输入特征矩阵的深度 输出特征矩阵的深度(主分支卷积核的个数) 步长 下采样参数(捷径中的1*1卷积层)
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 首先定义残存结构中要使用的一系列层结构      深度一致                            卷积核大小3*3
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)  # 不使用偏置
        # 定义BN层
        self.bn1 = nn.BatchNorm2d(out_channel)
        # 定义激活函数
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # 定义下采样方法就是我们传入的下采样方法-None
        self.downsample = downsample

    # 定义正向传播过程
    def forward(self, x):   # 输入特征矩阵x
        identity = x      # 将x赋值给identity-捷径上的输出值
        if self.downsample is not None:   # 如果没有传入下采样函数，对应图中实线的残差结构，即跳出
            identity = self.downsample(x)  # 如果有下采样函数，即将x传入下采样函数中

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)  # 不经过激活函数

        out+=identity     # 将输出加上捷径的输出再经过激活函数
        out=self.relu(out)  # 得到残差结构的最终输出

        return out

# 定义一个类-50，101，102层的残差结构
class Bottleneck(nn.Module):
    expansion = 4   # 反映卷积层使用卷积核个数的变化 4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1,bias=False)
                                # 1*1卷积层 步长1 通过卷积层1，特征矩阵的高和宽是不会发生变化的
        self.bn1 = nn.BatchNorm2d(out_channel)
                                                                                # 卷积核的个数是前面的4倍
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                               kernel_size=3, stride=stride, padding=1, bias=False)
                                        # 卷积层步距不再为1
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=1, stride=stride,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)  # 输出矩阵深度为卷积层三输出特征矩阵的深度
        self.relu = nn.ReLU(inplace=True)   # 定义激活函数
        self.downsample = downsample    # 下采样参数

    # 定义正向传播过程
    def forward(self, x):
        identity = x
        if self.downsample is not None:# 如果没有传入下采样函数，对应图中实线的残差结构，即跳出
            identity = self.downsample(x)  # 如果有下采样函数，即将x传入下采样函数中

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)

        out+=identity
        out=self.relu(out)

        return out

# 定义ResNet网络结构
class ResNet(nn.Module):
# block对应残差结构,会根据我们定义的层结构传入不同的block,num_block对应使用残差结构数量,列表参数
    def __init__(self, block, num_blocks, num_classes=1000,include_top=True):
        # 训练集的分类个数     # 方便在resnet网络基础上搭建搭建更复杂的网络
        super(ResNet, self).__init__()
        self.include_top = include_top    # 将include_top传入到我们的类变量中
        self.in_channel = 64     # 输入特征矩阵的深度-通过max_pooling之后得到的特征矩阵深度

# 构建ResNet中的第一个卷积层 7*7卷积层 输入特征矩阵深度-RGB          卷积核大小
        self.conv1=nn.Conv2d(3,self.in_channel,kernel_size=7,stride=2,
                             padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(self.in_channel)   # BN层 输入深度是特征层1的输出深度
        self.relu=nn.ReLU(inplace=True)   # 激活函数
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # 最大池化层
        self.layer1=self._make_layer(block, 64, num_blocks[0], stride=1)  # 对应表格中conv2_x的一系列残差结构-通过make_layer这个函数生成
        self.layer2=self._make_layer(block, 128, num_blocks[1], stride=2) # 对应表格中conv3_x的一系列残差结构
        self.layer3=self._make_layer(block, 256, num_blocks[2], stride=2) # .....
        self.layer4=self._make_layer(block, 512, num_blocks[3], stride=2)
        if self.include_top:  # 默认为True
            # 平均池化采样层    # 自适应-经过该采样层得到的特征矩阵高和宽都是1
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            # 通过全连接层-输出节点层 #展平后的节点个数
            self.fc = nn.Linear(512*block.expansion, num_classes) # 输出节点个数为分类类别个数

        # 对卷积层进行初始化操作
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')

    # 定义make_layer函数  Basicblock或Bottleneck
    def _make_layer(self, block, out_channels, num_blocks, stride): # out_channels-残差结构中卷积层第一次层所使用卷积核的个数

        #                                  # 该层一共包含多少个残差结构
        downsample = None
         #   18和34层的网络结构会跳过if语句
        if stride != 1 or self.in_channel != out_channels*block.expansion:
            # 生成下采样函数
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channels*block.expansion, kernel_size=1, stride=stride, bias=False),
                                          # 特征矩阵深度*4倍，高和宽不变
                nn.BatchNorm2d(out_channels*block.expansion))
        # 定义空列表
        layers=[]   # 第一层残差结构添加进去 输入特征矩阵的值 第一个卷积层的卷积核个数
        layers.append(block(self.in_channel, out_channels, stride=stride, downsample=downsample))
        self.in_channel = out_channels*block.expansion  # 深度发生变化

        # 第一层已经搭建好了，再通过循环将剩下一系列的实线部分的残差结构，压入
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channel, out_channels))
                            # 传入输入特征矩阵深度  残差结构主分支的一层卷积层的卷积核个数
                # 通过(*)非关键参数的形式传入到nn点.Sequential函数
        return nn.Sequential(*layers)  # 将定义的一系列层结构组合到一起返回

    # 正向传播过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # 3*3 max_pooling

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x=torch.flatten(x,1)   # 展平处理
            x = self.fc(x)         # 全连接

        return x     # 最终输出

# 定义34层resnet网络
def resnet34(num_classes=1000,include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, include_top)
   # 调用定义好的ResNet类 输入的block就是对应的残差结构 残差结构个数分别是 分类类别个数

# 定义r101层resnet网络
def resnet101(num_classes=1000,include_top=True):
    return ResNet(BasicBlock, [3, 4, 23, 3], num_classes, include_top)


