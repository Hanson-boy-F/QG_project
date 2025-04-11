# predict模型
import os
import json
import torch
from  PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from ResNet.train import net
from model import resnet34
import logging # 添加日志记录
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示正常负号

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 数据预处理要和训练集一致
    data_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                      [0.2023, 0.1994, 0.2010])])
    # 加载类别标签
    json_path = './class_indices.json'
    assert os.path.exists(json_path), 'json_path does not exist!'
    with open(json_path, 'r') as f:
        class_indict = json.load(f)

        # 加载模型权重
    weights_path = './resNet34.pth'
    assert os.path.exists(weights_path), f"文件 {weights_path} 不存在"
    net.load_state_dict(torch.load(weights_path, map_location=device))

    # 预测模式
    net.eval()

    while True:
        # 用户输入路径
        img_path = input("输入图像路径（输入r退出）: ")
        if img_path == "r":
            break

        assert os.path.exists(img_path), f"文件 {img_path} 不存在"

        # 加载图像
        img = Image.open(img_path).convert('RGB')   # 强制转换图片类型
        plt.imshow(img)

        # 预处理
        img_tensor = data_transforms(img)
        img_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

        # 执行预测
        with torch.no_grad():
            output = torch.squeeze(net(img_tensor))
            predict = torch.softmax(output, dim=0)
            pred_class = torch.argmax(predict).item()

        # 显示结果
        print_res = f"预测类别: {class_indict[str(pred_class)]} | 准确率: {predict[pred_class].item():.3f}"
        plt.title(print_res)
        plt.show()

        # 打印所有类别概率
        print("\n所有类别概率:")
        for i in range(len(class_indict)):
            print(f"{class_indict[str(i)]:10}: {predict[i].item():.3f}")
        print("------------------------")

        logging.basicConfig(filename='predict.log', level=logging.INFO)
        logging.info(f"图像: {img_path} | 预测结果: {class_indict[str(pred_class)]}")


if __name__ == '__main__':
    predict()
