# Deep Learning

## 深度学习播放列表概述 & 机器学习简介

Machine learning：机器学习是使用算法分析数据、==从数据中学习==，然后对新数据做出判断或预测的实践。

Deep learning：深度学习是一种可以用来实现机器学习的工具或技术。 

==从数据中学习==：机器学习中，我们不是手动编写具有特定指令的代码来完成特定任务，而是使用数据和算法来==训练==机器，使其能够执行任务

例如：Analyzing the sentiment of a popular media outlet and classifying that sentiment as positive or negative.

**传统方法：**

算法可能首先寻找与负面或正面情绪相关的特定词汇。

通过条件语句，算法将根据它所知道的正面或负面词汇将文章分类为正面或负面。



**机械学习方法：**

机械学习算法分析给定的媒体数据，并学习将负面文章与正面文章区分开来的特征。基于所学到的知识和大量的训练，这样算法可以就对新文章进行分类，判断其是正面还是负面

 

## 深度学习解析

  深度学习是机器学习的一个子领域，它使用受大脑神经网络结构和功能启发的算法,使用从数据中学习的算法。

  这种学习将以监督或无监督的形式进行

  监督学习基本上发生在深度学习模型的学习并从已经被标记的数据中进行推断中时

  无监督模型发生在模型学习并从未标记的数据中进行推断时



例如：一个深度学习模型，其目的是识别图像是猫还是狗

假设，我们把10000张猫和狗的图像提供给该学习模型进行学习，如果模型以监督形式学习，那么每张图像都会被标记为猫或狗，然后模型将了解哪张图像被标记了什么，然后学习猫狗之间的相似之处和不同之处；如果模型在无监督的形式下学习，那么图像就不会被标记为猫或狗，模型本质上是会学习不同图像的特征然后根据它们的相似性或差异将图像分类而无需再次知道或看到它们的标记。

![image-20250325134223105](D:\Markdown文件\神经网络.png)

​															(神经网络结构)

神经网络中的神经元被组织成我们所说的层。

层 *在*神经网络中（除了输入层和输出层之外）被称为隐藏层

如果一个神经网络有多个隐藏层，那么这个神经网络就被称作深度神经网络

![image-20250325134729561](C:\Users\Ye\AppData\Roaming\Typora\typora-user-images\image-20250325134729561.png)

总的来说，深度学习使用具有多个隐藏层的神经网络



## 神经网络解析

  人工神经网络是受大脑神经网络启发的计算系统，是一种由我们称之为层的一组相互连接的单元（称为神经元）组成的计算系统。这些神经元之间的每个连接都可以将信号从一个神经元传到另一个神经元，然后接收神经元处理该信号，然后向与其连接的下游神经元发送信号。

   神经元是分层组织的，不同层对其输入执行不同类型的转换。数据从==输入层==开始，通过==隐藏层==流动，直到达到==输出层==。这被称为网络的前向传播。位于输入层和输出层之间的层被称为隐藏层。

**可视化神经网络**

![image-20250326110758767](C:/Users/Ye/AppData/Roaming/Typora/typora-user-images/image-20250326110758767.png)

使用名为Keras的神经网络API,构建顺序模型

在 Keras 中，我们可以构建所谓的顺序模型。Keras 将顺序模型定义为线性层的顺序堆叠

```
#导入keras库
from keras.models import Sequential
from keras.layers import Dense,Activation

#构造函数传递一个数组,实际上，这些数组是Dense层
layers = [
    Dense(units=3, input_shape=(2,), activation='relu'),
    Dense(units=2, activation='softmax')
]
#建一个名为 model 的变量，并将其设置为 Sequential 对象的一个实例
model=Sequential(layers)

# input_shape=(2,) 表示输入层有2个神经元
#激活函数('relu','softmax')通常是遵循Dense层的非密集函数

#Dense层是神经网络中最基本的图层类型，它将每个输入连接到图层内的每个输出
#Hidden层被视为一个Dense层，它将每个输入连接到图层内的每个输出

```



## 神经网络中层的解析

 人工神经网络中的神经元通常按层组织，并且有许多不同类型的层，常用Dense层

- Dense (or fully connected) layers

- Convolutional layers

- Pooling layers

- Recurrent layers

- Normalization layers

	神经网络中的不同层对输入执行不同的转换，有些层更适合某些任务

Convolutional layers：用于处理图像数据的模型

Recurrent layers：用于处理时间序列数据的模型

Dense layers ：在其层内，将每个输入完全连接到每个输出

 

![image-20250326135803220](C:/Users/Ye/AppData/Roaming/Typora/typora-user-images/image-20250326135803220.png)

每个节点之间的连接都有一个相关的权重，每个权重代表两个节点之间连接的强度

当网络在输入层的某个节点接收到输入时，这个输入将通过连接传递到下一个节点，并且输入将被乘以分配给该连接的权重，对于第二层中的每个节点，都会计算每个输入连接的加权总和。然后将这个总和传递给一个激活函数，该函数会对给定的总和进行某种类型的转换

**节点输出 = 输入的加权和的激活**

一旦我们获得给定节点的输出，所获得的输出就是作为下一层节点输入的值。这个过程会一直持续到达到输出层。输出层中的节点数量取决于我们拥有的可能输出或预测类别数量

```
#导入keras库
from keras.models import Sequential
from keras.layers import Dense,Activation

# 数组形式传入函数
model=Sequential([
		Dense(5,input_shape=(3,),activation='relu')
		Dense(2,activation='softmax')
])
```

```
# 创建精确神经网络
# 创建一个具有5个神经元的Dense层代表Hidden layer
# 从第一个Dense层开始创建

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
%matplotlib inline

img=np.expand_dims(ndimage.imread('NN.png'),0)
plt.imshow(ing(0))


```

第一个 `Dense` 层是第一个隐藏层

一个名为 relu `activation='relu'` 的激活函数来处理我们的两个隐藏层，以及一个名为 softmax `activation='softmax'` 的激活函数来处理输出层

```
from keras.models import Sequential
from keras.layers import Dense, Activation

layers = [
    Dense(units=6, input_shape=(8,), activation='relu'),
    Dense(units=6, activation='relu'),
    Dense(units=4, activation='softmax')
]

model = Sequential(layers)
```

![image-20250326164143438](C:/Users/Ye/AppData/Roaming/Typora/typora-user-images/image-20250326164143438.png)

## 神经网络中的激活函数详解

激活函数:在人工神经网络中，激活函数是一个将节点的输入映射到其对应输出的函数

取了每个节点在层中每个输入连接的加权求和，并将其传递给激活函数

​							节点输出 = 激活（输入加权求和）

激活函数会对求和结果进行某种类型的操作，将其转换为通常介于某个下限和某个上限之间的数字

**Sigmoid 激活函数**

sigmoid 接收输入并执行以下操作:

- 对于大多数负输入，sigmoid 函数会将输入转换为非常接近于 0 的数字。
- 对于大多数正输入，sigmoid 函数会将输入转换为非常接近于 1 的数字。
- 对于相对接近 0 的输入，sigmoid 函数将输入转换为介于 0 和 1 之间的某个数字



![image-20250326220533416](C:/Users/Ye/AppData/Roaming/Typora/typora-user-images/image-20250326220533416.png)

使用人工神经网络中的 Sigmoid 激活函数，我们了解到神经元可以在 0 和 1 之间，越接近 1 ，该神经元的激活程度越高，而越接近 0 ，该神经元的激活程度越低

**ReLU 激活函数**

我们的激活函数并非总是会对输入进行变换，使其在 0 和 1 之间

ReLU，即*rectified linear unit*(修正线性单元)的缩写，将输入变换为 0#和输入本身中的最大值

​												**relu(x)= max(0,x)**

所以如果输入小于或等于 0 ，那么 relu 将输出 0 。如果输入大于 0 ，relu 将直接输出给定的输入,即神经元越积极，它的激活程度就越高。

**理解线性函数**

假设 f 是集合 X 上的一个函数。
假设 a 和 b 在 X 中。
假设 x 是一个实数。

如果f(x)是线性函数，则 ***f(a+b)=f(a)+f(b) ***,且 ***f(xa)=xf(a)***

线性函数的一个重要特性是两个线性函数的复合仍然是线性函数。这意味着，即使在非常深的神经网络中，如果我们只在正向传播过程中对我们的数据值进行线性变换，那么我们网络中从输入到输出的学习映射也将是线性的.

深度神经网络学习的映射类型比简单的线性映射更复杂,因此大多数激活函数都是非线性的，而且故意这样选择。拥有非线性激活函数使得我们的神经网络能够计算任意复杂的函数。

```
# 导入库
from keras.models import Sequential
from keras.layers import Dense,Activation
```

```
指定激活函数的第一种方式是在层的构造函数中
model=Sequential([
	Dense(units=5,input_shape(3,),activation='relu')
])
#在这种情况下，我们有一个 Dense 层，并指定 relu 作为我们的激活函数 activation='relu'
```

```
# 添加层和激活函数
model=Sequential()
model.add(Dense(units=5,input_shape(3,)))
model.add(Activation('relu'))
```

由于:

​					**节点输出 = 激活（输入加权求和）**

即ReLU作为激活函数

​					**节点输出 = ReLU（输入的加权和）**



## 训练一个神经网络解析

***What is training？***

当我们训练一个模型时，我们基本上是在尝试解决一个优化问题。我们试图优化模型中的权重。我们的任务是找到最准确地映射我们的输入数据到正确输出类别的权重。在训练过程中，使权重会迭代更新，并逐渐接近最优值。

将所有数据通过我们的模型后，我们将继续反复将相同的数据传递给模型。这种反复将相同数据传递给模型的过程被认为是 训练

 **优化算法**

权重是通过我们所说的优化算法进行优化的。优化过程取决于所选择的优化算法，最广为人知的优化算法被称为 *随机梯度下降算法*  即SGD

SGD 的目标是最小化损失函数,如MES，RMES，等，损失是网络预测情况与真实情况之间的误差或差异，SGD 将尝试最小化这个误差，使我们的模型在预测上尽可能准确。



## 神经网络学习过程解析

***epoch***

训练过程，每个用于训练的数据点都会通过模型，这种从输入到输出的网络传递称为前向传递，而结果输出取决于模型中每个连接的权重。当我们的数据集中的所有数据点都通过网络后，那么一个 *epoch* 就完成了

 *epoch* 指的是在训练过程中，模型对整个数据集的单次遍历。

模型初始化时，模型各层的输出权重被设置为任意值。在模型输出端，模型将为给定输入提供输出，一旦得到输出，可以通过比较模型预测值与真实值来计算特定输出的损失。

**损失函数的梯度**

在损失计算之后，将计算该损失函数相对于网络中每个权重的梯度，梯度是多个变量函数的导数一旦我们得到损失函数梯度的值，我们就可以使用这个值来更新模型的权重。梯度告诉我们哪个方向会使损失趋向最小值，我们的模型是朝着降低损失并接近这个最小值的方向前进。

**学习率**

我们将梯度值乘以一个称为学习率的数值，学习率告诉我们应该朝着最小值方向迈出多大的一步，也即是步长

**更新权重**

把梯度与学习率相乘，然后权重减去梯度与学习率的乘积，得出这个权重的新的更新值

​					*new weight= old weight -(learning rate \* gradient)*

所有这些权重在每个 epoch 中都会迭代更新。随着 SGD(随机梯度下降)实现最小化损失函数，权重将逐步接近其优化值。

这种权重的更新实际上就是我们所说的**模型学习的过程**，随着权重的变化，模型在准确地将输入映射到正确输出方面变得越来越聪明。

**Keras进行代码训练**

```
# 导入库
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
```

```
# 定义模型
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='sigmoid')
])
```

```
# 编译
# compile() 函数传递优化算法
model.compile(
    optimizer=Adam(learning_rate=0.0001), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)
```

```
# 将数据拟合到训练模型
model.fit(
    x=scaled_train_samples, #是一个包含训练样本的numpy数组
    y=train_labels, 		# 是一个包含训练样本对应标签的numpy数组
    batch_size=10, 			# 发送数据数量
    epochs=20, 				# 遍历次数
    shuffle=True, 			# 在传递给模型之前应先对数据进行打乱
    verbose=2
)
```

## “损失”在神经网络中的解析

**损失(*error*)**

在训练过程的每个 epoch 结束时，将使用模型的输出预测和相应的真实标签来计算损失。

假设我们的模型正在对猫和狗的图像进行分类，假设猫的标签是 0 ，狗的标签是 1 。

- 猫: 0
- 狗: 1

现在假设我们向模型传递一张猫的图像，并且模型提供的输出是 0.25 。在这种情况下，模型预测与真实标签之间的差异是 0.25−0.00=0.25 。这个差异也被称为误差。

error=0.25−0.00=0.25

这个过程对每个输出都进行。对于每个 epoch，错误会在所有单个输出上累积。

**均方误差（MSE）**

​        			***MSE(input)=1/n\*(output-label)(output- label)***

一次性将多个样本传递给模型（一个样本批次），那么我们将取所有这些样本的平方误差的平均值

每次更新权重时损失值都会发生变化,随着运行多个 ***epoch***，我们的损失值会降低

**损失函数在 Keras 中的代码**

```
# 导入库
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
```

```
# 定义模型
model = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='sigmoid')
])
```

```
model.compile(
    Adam(learning_rate=.0001),   
    loss='sparse_categorical_crossentropy',    # 指定的损失函数-稀疏分类交叉熵的损失函数
    metrics=['accuracy']  
)
```



## 学习率在神经网络中的解释

神经网络训练过程中训练是为了 SGD 最小化实际输出和从我们的训练样本中预测的输出之间的损失，为了达到最小损失所采取的步长将取决于学习率

学习率主要用于更新权重，一般在`0.01` 和 `0.0001` 之间，当将学习率设置在这个范围的较高数值时，可能会超出想要的目标。这是因为采取的步子太大，导致在最小化损失函数的方向上超过了这个最小值。

一般定义变量名为*learning_rate*



## 训练集、测试集和验证集的解析

**训练集**

训练集：用于训练模型的数据集，模型将使用这些训练集中的数据反复训练，并不断学习这些数据的特征

训练好的模型能根据从训练集中学到的知识来准确地预测从未见过的新数据

  **验证集**

独立于训练集的数据，用于在训练期间验证我们的模型，这个验证过程有助于我们调整超参数

训练的每个 epoch 中，模型将在训练集的数据上进行训练,还将同时在验证集的数据上进行验证

验证集的数据与训练集的数据是分开的,所以当模型在这组数据上进行验证时，这组数据并不包含模型在训练中已经熟悉的样本

如果我们也在验证集上验证模型，并且看到模型对验证数据的输出结果与对训练数据的输出结果一样好，那么我们可以更有信心地认为我们的模型没有过拟合；另一方面，如果训练数据上的结果非常好，但验证数据上的结果落后，那么我们的模型就是过拟合了。

验证集使得我们能够知道模型在训练过程中的泛化能力如何

**测试集**

测试集是在模型训练完成后用于测试模型的数据集。测试集与训练集和验证集都分开

测试集与另外两个集的最大区别在于，测试集不应该被标记。训练集和验证集必须被标记，这样我们才能看到训练期间给出的指标，如每个 *epoch* 的损失和准确率。

测试集提供了在将模型部署到生产之前，对模型泛化能力进行最终检查的功能

**机器学习和深度学习的最终目标是构建能够良好泛化的模型**

