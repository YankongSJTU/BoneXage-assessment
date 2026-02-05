#!/usr/bin/env python3
import os
import sys


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tqdm import tqdm
import warnings
import time
from datetime import datetime
import json
from pathlib import Path
import math
warnings.filterwarnings('ignore')
# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class Config:
    # 训练数据路径
    TRAIN_IMG_DIR = 'data/boneage-training-dataset/boneage-training-dataset/'
    TRAIN_CSV = 'data/boneage-training-dataset.csv'
    
    # 测试数据路径
    TEST_IMG_DIR = 'data/boneage-test-dataset/boneage-test-dataset/'
    TEST_CSV = 'data/boneage-test-dataset.csv'
    
    # 模型参数
    BATCH_SIZE = 80  # 每个GPU的batch size，因为图像尺寸变大了
    LEARNING_RATE = 0.0005
    EPOCHS = 200
    IMG_SIZE = 512  # 增加图像尺寸以提高精度
    
    # 训练参数
    LOG_INTERVAL = 10  # 每10个batch打印一次信息
    SAVE_INTERVAL = 5  # 每5个epoch保存一次中间模型
    
    # 多GPU参数
    NUM_GPUS = 4  # 使用的GPU数量
    
    # 优化器参数
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0  # 梯度裁剪
    
    # 学习率调度
    WARMUP_EPOCHS = 5
    COSINE_T_MAX = 50
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # 增加一点点随机裁剪，模拟位置偏移
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # 平移
        transforms.ToTensor(),
        # 建议使用 0.5 的均值标准差，因为这更接近 X 光的灰度分布
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
# ==================== 简化的先进模型架构 ====================

#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

# ==================== 核心组件 ====================

class GeM(nn.Module):
    """
    广义平均池化 (Generalized Mean Pooling)
    可学习的池化层，能够自动学习关注显著特征（介于 MaxPool 和 AvgPool 之间）
    对于医学图像中的微小骨化中心特征非常有效。
    """
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p是一个可学习的参数
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W) -> (B, C)
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p).flatten(1)

    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'

class SexEncoder(nn.Module):
    """处理性别信息的轻量级模块"""
    def __init__(self, output_dim=32):
        super(SexEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 确保输入维度正确 (B, 1)
        return self.fc(x.view(-1, 1))

# ==================== 主模型架构 ====================

class BoneAgeModel(nn.Module):
    def __init__(self, backbone_name='convnext_tiny', pretrained=True, max_age=240):
        """
        Args:
            backbone_name: 'convnext_tiny' 或 'resnet50'
            max_age: 骨龄的最大值(月)，用于缩放输出
        """
        super(BoneAgeModel, self).__init__()
        self.max_age = max_age
        
        # 1. 骨干网络 (推荐使用 ConvNeXt)
        if 'convnext' in backbone_name:
            # ConvNeXt-Tiny: 精度高，计算量适中，非常适合医学图像
            base_model = models.convnext_tiny(weights='DEFAULT' if pretrained else None)
            self.features = base_model.features
            # ConvNeXt-Tiny 最后的通道数是 768
            self.num_features = 768 
        elif 'resnet' in backbone_name:
            base_model = models.resnet50(weights='DEFAULT' if pretrained else None)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            self.num_features = 2048
        elif 'efficientnet' in backbone_name:
            base_model = models.efficientnet_v2_m(weights='DEFAULT' if pretrained else None)
            self.features = base_model.features
            self.num_features = 1280
        else:
            raise ValueError(f"Backbone {backbone_name} not supported")

        # 2. 池化层 (使用 GeM 替代简单的 AvgPool)
        self.pool = GeM()
        
        # 3. 性别编码
        self.sex_encoder = SexEncoder(output_dim=32)
        
        # 4. 回归头 (Regression Head)
        # 输入维度 = 图像特征维度 + 性别特征维度(32)
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.num_features + 32),
            nn.Dropout(0.5),
            nn.Linear(self.num_features + 32, 512),
            nn.Mish(inplace=True), # Mish 激活函数比 ReLU 更平滑
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 64),
            nn.Mish(inplace=True),
            nn.Linear(64, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, image, sex):
        # 1. 提取图像特征
        x = self.features(image) # (B, C, H, W)
        
        # 2. 池化
        x = self.pool(x) # (B, C)
        
        # 3. 处理性别
        s = self.sex_encoder(sex) # (B, 32)
        
        # 4. 特征融合
        feat = torch.cat([x, s], dim=1)
        
        # 5. 预测
        out = self.head(feat)
        
        # 6. 关键技巧：Range Scaling
        # 使用 Sigmoid 将输出压缩到 (0, 1)，然后乘最大骨龄
        # 这避免了模型训练初期预测出负数或极大值，极大稳定了 Loss
        return torch.sigmoid(out).squeeze() * self.max_age

# ==================== 损失函数 ====================

class L1_L2_Loss(nn.Module):
    """
    结合 L1 (MAE) 和 MSE 的损失函数。
    训练初期 MSE 占主导加速收敛，后期 L1 占主导提高精度。
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha # L1 的权重
        
    def forward(self, pred, target):
        loss_l1 = F.l1_loss(pred, target)
        loss_mse = F.mse_loss(pred, target)
        return self.alpha * loss_l1 + (1 - self.alpha) * loss_mse
class TrainingLogger:
    def __init__(self, log_file='checkpoints/trainlog.csv'):
        self.log_file = log_file
        self.log_data = []
        self.columns = [
            'timestamp', 'epoch', 'batch', 'phase',
            'loss', 'mae', 'rmse', 'r2', 'lr',
            'batch_time', 'data_time', 'samples_per_sec'
        ]
        if not os.path.exists(log_file):
            pd.DataFrame(columns=self.columns).to_csv(log_file, index=False)
    def log_batch(self, epoch, batch_idx, phase, loss, mae, rmse, r2, lr, batch_time, data_time, samples_per_sec):
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epoch': epoch,
            'batch': batch_idx,
            'phase': phase,
            'loss': float(loss),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2) if r2 is not None else None,
            'lr': float(lr),
            'batch_time': float(batch_time),
            'data_time': float(data_time),
            'samples_per_sec': float(samples_per_sec)
        }
        self.log_data.append(log_entry)
        if len(self.log_data) >= 10:
            self.flush()
    def log_epoch(self, epoch, phase, metrics, lr, epoch_time):
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epoch': epoch,
            'batch': -1,
            'phase': f'{phase}_epoch',
            'loss': float(metrics['loss']),
            'mae': float(metrics['mae']),
            'rmse': float(metrics['rmse']),
            'r2': float(metrics['r2']) if 'r2' in metrics else None,
            'lr': float(lr),
            'batch_time': 0.0,
            'data_time': 0.0,
            'samples_per_sec': float(epoch_time)
        }
        self.log_data.append(log_entry)
        self.flush()
    def flush(self):
        if self.log_data:
            df = pd.DataFrame(self.log_data)
            df.to_csv(self.log_file, mode='a', header=False, index=False)
            self.log_data = []

class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class BoneAgeDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None, is_test=False):
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

        # 读取CSV文件
        self.df = pd.read_csv(csv_path)

        # 如果是训练数据，需要处理boneage和male列
        if not is_test:
            if 'id' in self.df.columns:
                self.df = self.df.rename(columns={'id': 'Case ID'})

            if 'male' in self.df.columns:
                self.df['Sex'] = self.df['male'].apply(lambda x: 'M' if x else 'F')
                self.df = self.df.drop(columns=['male'])

            self.df['boneage'] = pd.to_numeric(self.df['boneage'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        case_id = str(self.df.iloc[idx]['Case ID'])
        img_path = os.path.join(self.img_dir, f"{case_id}.png")

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # 创建空白图像
            image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), color='black')

        if self.transform:
            image = self.transform(image)

        if not self.is_test:
            boneage = self.df.iloc[idx]['boneage']
            sex = 1.0 if self.df.iloc[idx]['Sex'] == 'M' else 0.0

            return {
                'image': image,
                'boneage': torch.tensor(boneage, dtype=torch.float32),
                'sex': torch.tensor(sex, dtype=torch.float32).unsqueeze(0)
            }
        else:
            sex = 1.0 if self.df.iloc[idx]['Sex'] == 'M' else 0.0
            return {
                'image': image,
                'sex': torch.tensor(sex, dtype=torch.float32).unsqueeze(0),
                'case_id': case_id
            }


class HybridLoss(nn.Module):
    """混合损失函数"""
    def __init__(self, alpha=0.7, beta=0.3):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target):
        # MAE损失
        mae_loss = F.l1_loss(input, target)

        # 相对误差损失
        rel_error = torch.abs((input - target) / (target + 1e-8))
        rel_loss = rel_error.mean()

        # 组合损失
        total_loss = self.alpha * mae_loss + self.beta * rel_loss

        return total_loss

