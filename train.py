#!/usr/bin/env python3
"""
手部X光骨龄预测训练脚本 - 修复版本
"""

import os
import sys

# 设置要使用的GPU设备
os.environ['CUDA_VISIBLE_DEVICES'] = "4,1,2,3"  # 使用4个GPU

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tqdm import tqdm
import warnings
from models import *
import time
from datetime import datetime
import json
from pathlib import Path
import math
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"可用的GPU数量: {gpu_count}")
    
    # 使用DataParallel进行多GPU训练
    device = torch.device('cuda')
    
    # 打印GPU信息
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"GPU {i}: {gpu_name}, 内存: {memory_total:.2f} GB")
else:
    device = torch.device('cpu')
    print("CUDA不可用，使用CPU")

print(f"Using device: {device}")

# 创建日志目录
log_dir = 'checkpoints'
os.makedirs(log_dir, exist_ok=True)
train_log_path = os.path.join(log_dir, 'trainlog.csv')


def train_epoch(model, dataloader, criterion, optimizer,scaler, device, epoch, logger, config):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    running_rmse = 0.0
    
    predictions = []
    targets = []
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    
    end = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        data_time.update(time.time() - end)
        
        images = batch['image'].to(device, non_blocking=True)
        boneages = batch['boneage'].to(device, non_blocking=True)
        sexes = batch['sex'].to(device, non_blocking=True)
        
        outputs = model(images, sexes)
        loss = criterion(outputs, boneages)
        
        optimizer.zero_grad()
        loss.backward()
        # ... 在 optimizer 定义之后 ...
        
        # 梯度裁剪
        if config.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        
        optimizer.step()
        
        batch_loss = loss.item()
        batch_preds = outputs.detach().cpu().numpy()
        batch_targets = boneages.cpu().numpy()
        
        batch_mae = mean_absolute_error(batch_targets, batch_preds)
        batch_rmse = np.sqrt(mean_squared_error(batch_targets, batch_preds))
        
        running_loss += batch_loss * images.size(0)
        running_mae += batch_mae * images.size(0)
        running_rmse += batch_rmse * images.size(0)
        
        predictions.extend(batch_preds)
        targets.extend(batch_targets)
        
        batch_time.update(time.time() - end)
        samples_per_sec = images.size(0) / batch_time.val
        
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            current_lr = optimizer.param_groups[0]['lr']
            
            try:
                batch_r2 = r2_score(batch_targets, batch_preds) if len(batch_targets) > 1 else 0.0
            except:
                batch_r2 = 0.0
            
            print(f'Epoch: [{epoch}][{batch_idx+1}/{len(dataloader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {batch_loss:.4f}\t'
                  f'MAE {batch_mae:.2f}\t'
                  f'RMSE {batch_rmse:.2f}\t'
                  f'R² {batch_r2:.4f}\t'
                  f'LR {current_lr:.6f}')
            
            logger.log_batch(
                epoch=epoch,
                batch_idx=batch_idx,
                phase='train',
                loss=batch_loss,
                mae=batch_mae,
                rmse=batch_rmse,
                r2=batch_r2,
                lr=current_lr,
                batch_time=batch_time.val,
                data_time=data_time.val,
                samples_per_sec=samples_per_sec
            )
        
        end = time.time()
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_mae = running_mae / len(dataloader.dataset)
    epoch_rmse = running_rmse / len(dataloader.dataset)
    epoch_r2 = r2_score(targets, predictions)
    
    return epoch_loss, epoch_mae, epoch_r2, predictions, targets

def validate_epoch(model, dataloader, criterion, device, epoch, logger, config):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    running_rmse = 0.0
    
    predictions = []
    targets = []
    
    batch_time = AverageMeter('Time', ':6.3f')
    end = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device, non_blocking=True)
            boneages = batch['boneage'].to(device, non_blocking=True)
            sexes = batch['sex'].to(device, non_blocking=True)
            
            outputs = model(images, sexes)
            loss = criterion(outputs, boneages)
            
            batch_loss = loss.item()
            batch_preds = outputs.cpu().numpy()
            batch_targets = boneages.cpu().numpy()
            
            batch_mae = mean_absolute_error(batch_targets, batch_preds)
            batch_rmse = np.sqrt(mean_squared_error(batch_targets, batch_preds))
            
            running_loss += batch_loss * images.size(0)
            running_mae += batch_mae * images.size(0)
            running_rmse += batch_rmse * images.size(0)
            
            predictions.extend(batch_preds)
            targets.extend(batch_targets)
            
            batch_time.update(time.time() - end)
            
            if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                try:
                    batch_r2 = r2_score(batch_targets, batch_preds) if len(batch_targets) > 1 else 0.0
                except:
                    batch_r2 = 0.0
                
                print(f'Val: [{epoch}][{batch_idx+1}/{len(dataloader)}]\t'
                      f'Time {batch_time.val:.3f}\t'
                      f'Loss {batch_loss:.4f}\t'
                      f'MAE {batch_mae:.2f}\t'
                      f'RMSE {batch_rmse:.2f}\t'
                      f'R² {batch_r2:.4f}')
            
            end = time.time()
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_mae = running_mae / len(dataloader.dataset)
    epoch_rmse = running_rmse / len(dataloader.dataset)
    epoch_r2 = r2_score(targets, predictions)
    
    return epoch_loss, epoch_mae, epoch_rmse, epoch_r2, predictions, targets

def save_checkpoint(state, filename, is_best=False):
    torch.save(state, filename)
    if is_best:
        best_filename = filename.replace('.pth', '_best.pth')
        torch.save(state, best_filename)

# ==================== 主训练流程 ====================

def main():
    config = Config()
    
    # 初始化日志记录器
    logger = TrainingLogger(train_log_path)
    
    print("=" * 80)
    print("开始训练手部X光骨龄预测模型")
    print(f"使用的GPU数量: {torch.cuda.device_count()}")
    print(f"训练日志将保存到: {train_log_path}")
    print("=" * 80)
    
    print("加载数据集...")
    
    # 加载训练数据集
    train_dataset = BoneAgeDataset(
        img_dir=config.TRAIN_IMG_DIR,
        csv_path=config.TRAIN_CSV,
        transform=config.train_transform,
        is_test=False
    )
    
    print(f"总样本数: {len(train_dataset)}")
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_dataset.dataset.transform = config.val_transform
    
    print(f"训练样本: {len(train_dataset)}")
    print(f"验证样本: {len(val_dataset)}")
    
    # 创建数据加载器
    total_batch_size = config.BATCH_SIZE * max(1, torch.cuda.device_count())
    print(f"总batch size: {total_batch_size}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=total_batch_size,
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=total_batch_size,
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型
    print("初始化模型...")
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load("./checkpoints/best_model.pth", map_location=device)
    # 在 main() 函数中
    model = BoneAgeModel(backbone_name='convnext_tiny', pretrained=True) # 使用新模型类
    #model = AdvancedBoneAgeModel( pretrained=True, use_efficientnet=True, use_attention=True)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]
        else:
            new_key = key
        new_state_dict[new_key] = value

    # 加载权重
    model.load_state_dict(new_state_dict, strict=True)
#    model.to(device)
 #   model.eval()

    # 使用DataParallel进行多GPU训练
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # 输出模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 损失函数
    criterion = HybridLoss(alpha=0.7, beta=0.3)
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scaler = GradScaler() # 初始化混合精度缩放器
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
        eta_min=config.LEARNING_RATE * 0.01
    )
    
    # 检查点
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    start_epoch = 0
    best_mae = float('inf')
    
    # 训练记录
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mae': [], 'val_mae': [],
        'train_r2': [], 'val_r2': [],
        'learning_rate': []
    }
    
    print(f"\n开始训练，总轮数: {config.EPOCHS}")
    print(f"图像尺寸: {config.IMG_SIZE}x{config.IMG_SIZE}")
    print("=" * 80)
    
    # 训练循环
    for epoch in range(start_epoch, config.EPOCHS):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print("-" * 80)
        
        # 训练
        train_loss, train_mae, train_r2, train_preds, train_targets = train_epoch(
            model, train_loader, criterion, optimizer, scaler,device, 
            epoch + 1, logger, config
        )
        
        # 验证
        val_loss, val_mae, val_rmse, val_r2, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device, epoch + 1, logger, config
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        epoch_time = time.time() - epoch_start_time
        
        # 记录epoch信息
        logger.log_epoch(
            epoch=epoch + 1,
            phase='train',
            metrics={'loss': train_loss, 'mae': train_mae, 'rmse': 0, 'r2': train_r2},
            lr=optimizer.param_groups[0]['lr'],
            epoch_time=epoch_time
        )
        
        logger.log_epoch(
            epoch=epoch + 1,
            phase='val',
            metrics={'loss': val_loss, 'mae': val_mae, 'rmse': val_rmse, 'r2': val_r2},
            lr=optimizer.param_groups[0]['lr'],
            epoch_time=epoch_time
        )
        
        # 打印epoch总结
        print(f"\nEpoch {epoch+1} 总结:")
        print(f"训练 - Loss: {train_loss:.4f}, MAE: {train_mae:.2f}, R²: {train_r2:.4f}")
        print(f"验证 - Loss: {val_loss:.4f}, MAE: {val_mae:.2f}, R²: {val_r2:.4f}, RMSE: {val_rmse:.2f}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}, 时间: {epoch_time:.2f}s")
        
        # 保存最佳模型
        if val_mae < best_mae:
            best_mae = val_mae
            
            # 如果是DataParallel，保存原始模型
            if isinstance(model, nn.DataParallel):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': best_mae,
                'val_loss': val_loss,
                'val_r2': val_r2,
                'train_mae': train_mae,
                'train_loss': train_loss
            }, os.path.join(checkpoint_dir, 'best_model.pth'), is_best=True)
            
            print(f"✅ 保存最佳模型，验证MAE: {best_mae:.2f} 个月 ({best_mae/12:.2f} 年)")
        
        # 定期保存检查点
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            if isinstance(model, nn.DataParallel):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'val_loss': val_loss,
                'val_r2': val_r2
            }, checkpoint_path)
            
            print(f"保存检查点到: {checkpoint_path}")
    
    print("\n" + "=" * 80)
    print("训练完成!")
    print(f"最佳验证MAE: {best_mae:.2f} 个月 ({best_mae/12:.2f} 年)")
    print("=" * 80)
    
    # 保存训练历史
    history_path = os.path.join(log_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
    print(f"训练历史已保存到: {history_path}")
    
    # 生成最终测试预测
    generate_test_predictions(model, device, config)
    
    return model, history, best_mae

def generate_test_predictions(model, device, config):
    """生成测试集预测结果"""
    print("\n生成测试集预测...")
    
    test_dataset = BoneAgeDataset(
        img_dir=config.TEST_IMG_DIR,
        csv_path=config.TEST_CSV,
        transform=config.val_transform,
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config().BATCH_SIZE * 4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    case_ids = []
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            images = batch['image'].to(device, non_blocking=True)
            sexes = batch['sex'].to(device, non_blocking=True)
            ids = batch['case_id']
            
            outputs = model(images, sexes)
            
            predictions.extend(outputs.cpu().numpy())
            case_ids.extend(ids)
    
    # 保存预测结果
    test_results = pd.DataFrame({
        'Case ID': case_ids,
        'Predicted_Bone_Age_Months': predictions,
        'Predicted_Bone_Age_Years': [p / 12.0 for p in predictions]
    })
    
    test_results['Formatted_Age'] = test_results['Predicted_Bone_Age_Months'].apply(
        lambda x: f"{int(x/12)}岁{int(x%12)}个月"
    )
    
    output_path = os.path.join(log_dir, 'test_predictions.csv')
    test_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"测试集预测结果已保存到: {output_path}")
    
    return test_results

# 运行主程序
if __name__ == "__main__":
    try:
        # 检查数据路径
        config = Config()
        
        required_paths = [
            (config.TRAIN_IMG_DIR, "训练图像目录"),
            (config.TRAIN_CSV, "训练CSV文件"),
        ]
        
        for path, description in required_paths:
            if not os.path.exists(path):
                print(f"⚠️  警告: {description}不存在: {path}")
        
        # 清空GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 训练模型
        trained_model, history, best_mae = main()
        
        print("\n" + "="*80)
        print("项目完成!")
        print("="*80)
        
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
