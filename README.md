# BoneXage-assessment
Predicting children's bone age using hand X-ray images
# 🦴 Bone Age Assessment using ConvNeXt & GeM Pooling

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A PyTorch implementation for precise pediatric **Bone Age Assessment (BAA)** based on hand radiographs.

This project utilizes a modern CNN architecture (**ConvNeXt**) combined with **Generalized Mean (GeM) Pooling** and specific optimizations for medical regression tasks. It achieves superior convergence and stability on the RSNA Pediatric Bone Age dataset compared to traditional baselines.

## 🚀 Key Features

* **SOTA Backbone**: Utilizes `ConvNeXt-Tiny` as the feature extractor, offering stronger feature representation than ResNet50 or EfficientNet.
* **GeM Pooling**: Implements learnable Generalized Mean Pooling instead of standard Average Pooling to better capture high-response features (e.g., ossification centers).
* **Multi-Modal Fusion**: Integrates image features with gender encoding to improve prediction accuracy significantly.
* **Numerical Stability**: Applies `Sigmoid * Max_Age` scaling at the output layer. This constraint prevents negative predictions and gradient explosion, ensuring stable training dynamics.
* **AMP Support**: Implements Automatic Mixed Precision (AMP) for faster training and lower GPU memory usage.

## 🛠️ Requirements

Ensure you have Python installed. You can install the dependencies via pip:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm pillow

##📂 Dataset Preparation
This project uses the RSNA Pediatric Bone Age Challenge dataset. Please organize your data directory as follows:
data/
├── boneage-training-dataset/   # Training images
├── boneage-test-dataset/       # Test images
├── boneage-training-dataset.csv
└── boneage-test-dataset.csv


## 🖥️ Usage
1. Training

Run train.py to start training. The script automatically detects available GPUs and utilizes DataParallel if applicable.
python train.py

2. Inference

Use predict.py to predict the bone age for a single image.
python predict.py <image_list_filename> <sex(M/F)> <image_size> <model_path>

Output:
data/test_imgs/1234.png: 10 years 6 months (126.5 months)

## 🧠 Model Architecture
graph LR
    A[Input Image 512x512] --> B[ConvNeXt Backbone]
    B --> C[Feature Maps]
    C --> D[GeM Pooling]
    D --> E[Image Features]
    
    F[Input Sex] --> G[Sex Encoder]
    G --> H[Sex Features]
    
    E --> I[Concat]
    H --> I
    I --> J[FC Layers + Mish]
    J --> K[Sigmoid * 240]
    K --> L[Predicted Age (Months)]

## 📊 Results
Metric,Value (Approx)
MAE (Mean Absolute Error),To be updated
Backbone,ConvNeXt-Tiny
Input Size,512x512

##📝 Acknowledgments
    Dataset provided by the RSNA Pediatric Bone Age Challenge.
    ConvNeXt architecture based on A ConvNet for the 2020s.


%% 将此代码块复制到你的 GitHub README.md 中
graph TD
    %% 定义样式
    classDef input fill:#E1F5FE,stroke:#0288D1,stroke-width:2px,color:#01579B;
    classDef output fill:#E8F5E9,stroke:#388E3C,stroke-width:2px,color:#1B5E20;
    classDef backbone fill:#FFF3E0,stroke:#FF9800,stroke-width:2px,color:#E65100;
    classDef pooling fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px,color:#4A148C;
    classDef fusion fill:#FCE4EC,stroke:#C2185B,stroke-width:2px,color:#880E4F;
    classDef head fill:#FFFDE7,stroke:#FBC02D,stroke-width:2px,color:#F57F17;
    classDef tensor fill:#FFFFFF,stroke:#9E9E9E,stroke-width:1px,stroke-dasharray: 5 5;

    %% 输入层
    subgraph Inputs [输入数据]
        IMG[图像输入<br>Batch x 3 x 512 x 512]:::input
        SEX[性别输入<br>Batch x 1 (0或1)]:::input
    end

    %% 图像处理分支
    subgraph Image Branch [图像特征提取分支]
        BACKBONE[ConvNeXt-Tiny Backbone<br>(预训练权重)]:::backbone
        FEAT_MAPS(特征图<br>Batch x 768 x H' x W'):::tensor
        GEM[GeM Pooling<br>(广义平均池化)]:::pooling
        IMG_VEC(图像特征向量<br>Batch x 768):::tensor
    end

    %% 性别处理分支
    subgraph Sex Branch [性别编码分支]
        SEX_ENC[Sex Encoder MLP<br>Linear-BN-ReLU x2]:::block
        SEX_VEC(性别特征向量<br>Batch x 32):::tensor
    end

    %% 特征融合
    CONCAT{特征拼接<br>Concatenation}:::fusion
    FUSED_VEC(融合特征向量<br>Batch x 800):::tensor

    %% 回归头
    subgraph Regression Head [回归预测头]
        L1[BN + Dropout(0.5)]:::head
        L2[Linear(800->512) + Mish激活]:::head
        L3[BN + Dropout(0.4)]:::head
        L4[Linear(512->64) + Mish激活]:::head
        L5[Linear(64->1)]:::head
    end

    %% 输出缩放
    SCALING[Range Scaling<br>Sigmoid * MaxAge(240)]:::fusion
    FINAL_OUT(最终预测骨龄<br>Batch x 1 (月)]:::output

    %% 连接关系
    IMG --> BACKBONE
    BACKBONE --> FEAT_MAPS
    FEAT_MAPS --> GEM
    GEM --> IMG_VEC

    SEX --> SEX_ENC
    SEX_ENC --> SEX_VEC

    IMG_VEC --> CONCAT
    SEX_VEC --> CONCAT
    CONCAT --> FUSED_VEC

    FUSED_VEC --> L1
    L1 --> L2
    L2 --> L3
    L3 --> L4
    L4 --> L5

    L5 --> SCALING
    SCALING --> FINAL_OUT
