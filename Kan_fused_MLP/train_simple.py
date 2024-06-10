import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from model.SimpleCNN import *
from model.KAN import KAN_Convolution_Network


def train(model, device, train_loader, optimizer, epoch, criterion):
    """
    训练模型一个epoch

    参数:
        model: 神经网络模型
        device: 使用的设备（cuda或cpu）
        train_loader: 训练数据的DataLoader
        optimizer: 优化器（例如SGD）
        epoch: 当前的epoch数
        criterion: 损失函数（例如CrossEntropy）

    返回:
        avg_loss: 当前epoch的平均损失
    """
    model.to(device)  # 将模型加载到指定设备上
    model.train()  # 设置模型为训练模式
    train_loss = 0  # 初始化训练损失
    # 分批处理图像
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)  # 将数据和目标加载到指定设备上
        optimizer.zero_grad()  # 重置优化器的梯度
        output = model(data)  # 将数据前向传递通过模型层
        loss = criterion(output, target)  # 计算损失
        train_loss += loss.item()  # 累加损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新参数

    avg_loss = train_loss / (batch_idx + 1)  # 计算当前epoch的平均损失
    return avg_loss


def test(model, device, test_loader, criterion):
    """
    测试模型

    参数:
        model: 神经网络模型
        device: 使用的设备（cuda或cpu）
        test_loader: 测试数据的DataLoader
        criterion: 损失函数（例如CrossEntropy）

    返回:
        test_loss: 测试集上的平均损失
        accuracy: 模型在测试集上的准确率
        precision: 模型在测试集上的精确率
        recall: 模型在测试集上的召回率
        f1: 模型在测试集上的F1分数
    """
    model.eval()  # 设置模型为评估模式
    test_loss = 0  # 初始化测试损失
    correct = 0  # 初始化正确预测计数
    all_targets = []  # 初始化所有真实标签列表
    all_predictions = []  # 初始化所有预测标签列表

    with torch.no_grad():  # 在不计算梯度的上下文中
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # 将数据和目标加载到指定设备上
            output = model(data)  # 获取预测结果
            test_loss += criterion(output, target).item()  # 计算并累加损失
            _, predicted = torch.max(output.data, 1)  # 获取预测的类别
            correct += (target == predicted).sum().item()  # 计算正确预测的数量
            all_targets.extend(target.view_as(predicted).cpu().numpy())  # 收集所有真实标签
            all_predictions.extend(predicted.cpu().numpy())  # 收集所有预测标签

    test_loss /= len(test_loader.dataset)  # 归一化测试损失
    accuracy = correct / len(test_loader.dataset)  # 计算准确率
    return test_loss, accuracy

def train_and_test_models(model, device, train_loader, test_loader, optimizer, criterion, epochs, scheduler):
    """
    训练并测试模型

    参数:
        model: 神经网络模型
        device: 使用的设备（cuda或cpu）
        train_loader: 训练数据的DataLoader
        test_loader: 测试数据的DataLoader
        optimizer: 优化器（例如SGD）
        criterion: 损失函数（例如CrossEntropy）
        epochs: 训练的epoch数
        scheduler: 学习率调度器

    返回:
        all_train_loss: 每个epoch的平均训练损失列表
        all_test_loss: 每个epoch的平均测试损失列表
        all_test_accuracy: 每个epoch的准确率列表
        all_test_precision: 每个epoch的精确率列表
        all_test_recall: 每个epoch的召回率列表
        all_test_f1: 每个epoch的F1分数列表
    """
    # 跟踪指标
    all_train_loss = []
    all_test_loss = []
    all_test_accuracy = []

    for epoch in range(1, epochs + 1):
        # 训练模型
        train_loss = train(model, device, train_loader, optimizer, epoch, criterion)
        all_train_loss.append(train_loss)
        # 测试模型
        test_loss, test_accuracy = test(model, device, test_loader, criterion)
        all_test_loss.append(test_loss)
        all_test_accuracy.append(test_accuracy)
        print(f'Epoch {epoch}: train_loss: {train_loss:.6f}, test_loss: {test_loss:.4f}, accuracy: {test_accuracy:.2%}')
        scheduler.step()
    model.all_test_accuracy = all_test_accuracy

    return all_train_loss, all_test_loss, all_test_accuracy

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize((0.5,), (0.5,))  # 归一化，将图像数据的每个通道归一化到均值为0.5，标准差为0.5
    ])
    # 加载MNIST训练集，并应用定义好的数据转换操作
    mnist_train = MNIST(root='./data', train=True, download=True, transform=transform)
    # 加载MNIST测试集，并应用定义好的数据转换操作
    mnist_test = MNIST(root='./data', train=False, download=True, transform=transform)
    # 定义训练数据的DataLoader，用于批量加载数据
    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    # 定义测试数据的DataLoader，用于批量加载数据
    test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_SimpleCNN = SimpleCNN()
    # 将模型移动到指定设备上
    model_SimpleCNN.to(device)
    # 定义优化器
    optimizer_SimpleCNN = optim.AdamW(model_SimpleCNN.parameters(), lr=1e-3, weight_decay=1e-4)
    # 定义学习率调度器
    scheduler_SimpleCNN = optim.lr_scheduler.ExponentialLR(optimizer_SimpleCNN, gamma=0.8)
    # 定义损失函数
    criterion_SimpleCNN = nn.CrossEntropyLoss()
    # 训练并测试简单CNN模型
    all_train_loss_SimpleCNN, all_test_loss_SimpleCNN, all_test_accuracy_SimpleCNN = train_and_test_models(
        model_SimpleCNN,
        device,
        train_loader,
        test_loader,
        optimizer_SimpleCNN,
        criterion_SimpleCNN,
        epochs=10,
        scheduler=scheduler_SimpleCNN
    )
    print("train_loss: ", all_train_loss_SimpleCNN)
    print("test_loss: ", all_test_loss_SimpleCNN)
    print("test_accuracy: ", all_test_accuracy_SimpleCNN)

if __name__ == "__main__":
    main()




