from torchvision import transforms
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm  # 进度条工具（可选，安装：pip install tqdm）
path_train = 'D:/BaiduNetdiskDownload/challenges-in-representation-learning-facial-expression-recognition-challenge/1/train'
path_vaild = 'D:/BaiduNetdiskDownload/challenges-in-representation-learning-facial-expression-recognition-challenge/1/val'

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0],-1)


transforms_train = transforms.Compose([
    transforms.Grayscale(),#使用ImageFolder默认扩展为三通道，重新变回去就行
    transforms.RandomHorizontalFlip(),#随机翻转
    transforms.ColorJitter(brightness=0.5, contrast=0.5),#随机调整亮度和对比度
    transforms.ToTensor()
])
transforms_vaild = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

data_train = torchvision.datasets.ImageFolder(root=path_train,transform=transforms_train)
data_vaild = torchvision.datasets.ImageFolder(root=path_vaild,transform=transforms_vaild)

train_set = torch.utils.data.DataLoader(dataset=data_train,batch_size=128,shuffle=True)
vaild_set = torch.utils.data.DataLoader(dataset=data_vaild,batch_size=128,shuffle=False)

def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 这里会使宽高减半
    return nn.Sequential(*blk)

def vgg(conv_arch, fc_features, fc_hidden_units):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # 每经过一个vgg_block都会使宽高减半
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_convs, in_channels, out_channels))
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

model = vgg(conv_arch, fc_features, fc_hidden_units)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 将模型移动到GPU（如果可用）
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

# 训练参数
num_epochs = 100
best_val_accuracy = 0.0

# 训练循环
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    with tqdm(train_set, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]', unit='batch') as t:
        for inputs, labels in t:
            inputs, labels = inputs.to(device), labels.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计信息
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # 更新进度条
            t.set_postfix(loss=loss.item(), acc=train_correct / train_total)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in vaild_set:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # 计算平均损失和准确率
    train_loss /= len(train_set)
    train_accuracy = train_correct / train_total
    val_loss /= len(vaild_set)
    val_accuracy = val_correct / val_total

    # 更新学习率
    scheduler.step(val_loss)

    # 打印epoch结果
    print(f'Epoch {epoch + 1}/{num_epochs}: '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

    # 保存最佳模型
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'New best model saved with val accuracy: {best_val_accuracy:.4f}')

print('Training complete!')
print(f'Best validation accuracy: {best_val_accuracy:.4f}')