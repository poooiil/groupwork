#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os  # 一定要记得导入 os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import joblib

# 定义一个简单的多层感知机模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class KeyPointClassifier:
    def __init__(self,
                 csv_path='keypoint_classifier_label.csv',
                 label_csv='keypoint_classifier_label.csv',
                 data_csv='keypoint.csv',
                 batch_size=32,
                 learning_rate=0.001,
                 num_epochs=20,
                 hidden_dim=100,
                 test_size=0.2,
                 random_state=42):
        # 获取当前脚本所在的绝对目录
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # 拼接文件的完整路径
        label_csv_path = os.path.join(base_dir, label_csv)
        data_csv_path  = os.path.join(base_dir, data_csv)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.hidden_dim = hidden_dim
        self.test_size = test_size
        self.random_state = random_state

        # 1. 读取 label 文件，构造映射
        label_mapping = pd.read_csv(label_csv_path, header=None)
        self.label_dict = {i: label_mapping.iloc[i, 0] for i in range(len(label_mapping))}
        self.labels = self.label_dict

        # 2. 读取训练数据 keypoint.csv
        data = pd.read_csv(data_csv_path)
        self.y = data.iloc[:, 0].values    # 第一列为标签
        self.X = data.iloc[:, 1:].values   # 其余列为特征

        # 3. 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

        # 4. 标准化
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # 5. 转换为 PyTorch 的 Tensor 并构造 DataLoader
        X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(self.y_test, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # 6. 构造模型
        self.input_dim = self.X_train.shape[1]  # 特征数
        self.num_classes = len(self.label_dict)   # 类别数
        self.model = MLP(self.input_dim, self.hidden_dim, self.num_classes)

        # 7. 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train_model(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(self.train_loader.dataset)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}")

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")

    def save_model(self, model_path='pytorch_mlp_model.pth', scaler_path='scaler.save'):
        torch.save(self.model.state_dict(), model_path)
        joblib.dump(self.scaler, scaler_path)
        print("Model and scaler saved.")

    def predict(self, landmark_list):
        """
        对输入的预处理关键点列表进行预测，返回预测类别的索引。
        假设 landmark_list 是一个一维列表或数组，其长度与训练时特征数一致。
        """
        import numpy as np
        # 将输入转换为二维数组（1, 特征数）
        landmark_array = np.array(landmark_list).reshape(1, -1)
        # 使用已经拟合好的 scaler 进行标准化
        landmark_array = self.scaler.transform(landmark_array)
        # 转换为 torch tensor
        input_tensor = torch.tensor(landmark_array, dtype=torch.float32)
        # 模型预测
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()

    def __call__(self, landmark_list):
        return self.predict(landmark_list)

if __name__ == '__main__':
    classifier = KeyPointClassifier()
    classifier.train_model()
    classifier.evaluate()
    classifier.save_model()
