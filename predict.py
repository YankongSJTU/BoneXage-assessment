
#!/usr/bin/env python3
"""
修复版骨龄预测脚本 - 与训练代码完全匹配
"""

import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "6"

import torch
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from torchvision import transforms
import torch.nn as nn
from models import *
from torchvision import models

def load_trained_model(model_path='checkpoints/best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model = BoneAgeModel(backbone_name='convnext_tiny', pretrained=True)
  #  model = AdvancedBoneAgeModel( pretrained=False, use_efficientnet=True, use_attention=True  )
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
    model.to(device)
    model.eval()
    return model, device
def predict_single_image(image_path, sex='M', image_size=1120, model_path='checkpoints/best_model.pth'):
    model, device = load_trained_model(model_path)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 编码性别
    sex_code = 1.0 if str(sex).upper() in ['M', 'MALE', '1'] else 0.0
    sex_tensor = torch.tensor([[sex_code]], dtype=torch.float32).to(device)
    
    # 预测
    with torch.no_grad():
        prediction = model(image_tensor, sex_tensor)
        bone_age = float(prediction.item())
    
    return bone_age

def main():
    # 简单的命令行参数处理
    if len(sys.argv) < 2:
        print("用法: python predict_fixed.py <图像路径> [性别=M] [图像尺寸=1120] [模型路径]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    sex = sys.argv[2] if len(sys.argv) > 2 else 'M'
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 1120
    model_path = sys.argv[4] if len(sys.argv) > 4 else 'checkpoints/best_model.pth'
    
    # 检查文件
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    try:
        for line in open(image_path):
            file2=line.rstrip()
            bone_age_months = predict_single_image(file2, sex, size, model_path)
            
            # 转换格式
            years = int(bone_age_months / 12)
            months = int(bone_age_months % 12)
            
            # 只输出最终结果
            print(f"{file2}: {years}岁{months}个月 ({bone_age_months:.1f}个月)")
        
    except Exception as e:
        print(f"预测错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
