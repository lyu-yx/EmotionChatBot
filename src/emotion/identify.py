import torch
import cv2
import time
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 这里会使宽高减半
    return nn.Sequential(*blk)


def vgg(conv_arch, fc_features, fc_hidden_units):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # 每经过一个vgg_block都会使宽高减半
        net.add_module("vgg_block_" + str(i + 1), vgg_block(num_convs, in_channels, out_channels))
    # 全连接层部分
    net.add_module("fc", nn.Sequential(
        Reshape(),
        nn.Linear(fc_features, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, 7)
    ))
    return net


# 修改conv_arch定义
conv_arch = ((1, 1, 32), (1, 32, 64), (2, 64, 128))  # 将第一个3改为1

# 其余代码保持不变
fc_features = 128 * 6 * 6  # 这个值可能需要调整，取决于你的输入尺寸
fc_hidden_units = 1024


def load_model(model_path, num_classes):
    model = vgg(conv_arch, fc_features, fc_hidden_units)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model


# 图像预处理 - 修改为转换为灰度图
def preprocess_image(image):
    # 将BGR转换为灰度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),  # 根据你的模型输入尺寸调整
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 灰度图只需要一个通道的均值和标准差
    ])
    return transform(gray_image).unsqueeze(0).to(device)


# 表情类别 (根据你的模型调整)
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# 主函数
def main():
    # 参数设置
    model_path = 'best_model.pth'
    num_classes = 7  # 根据你的模型调整
    interval = 0.1  # 检测间隔(秒)

    # 加载模型
    model = load_model(model_path, num_classes)
    print("模型加载完成，开始表情检测...")

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧")
            break

        # 显示实时画面
        cv2.imshow('Emotion Detection', frame)

        # 每5秒检测一次
        current_time = time.time()
        if current_time - last_time >= interval:
            try:
                # 预处理图像
                input_image = preprocess_image(frame)

                # 预测表情
                with torch.no_grad():
                    outputs = model(input_image)
                    # 计算softmax概率
                    probabilities = F.softmax(outputs, dim=1)
                    # 获取预测结果和概率
                    probs, predicted = torch.max(probabilities, 1)
                    emotion = emotion_classes[predicted.item()]

                    # 获取所有类别的概率
                    prob_list = probabilities.squeeze().cpu().numpy()
                    prob_dict = {emotion_classes[i]: f"{prob_list[i] * 100:.2f}%"
                                 for i in range(len(emotion_classes))}

                # 输出结果
                print(f"\n检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"预测表情: {emotion} (置信度: {probs.item() * 100:.2f}%)")
                print("各类别概率分布:")
                for emo, prob in prob_dict.items():
                    print(f"{emo}: {prob}")

            except Exception as e:
                print(f"检测出错: {str(e)}")

            # 重置计时器
            last_time = current_time

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    return probabilities

if __name__ == '__main__':
    softmax = main()
    print(softmax)