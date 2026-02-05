graph TD
    %% 定义样式
    classDef input fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1,rx:5,ry:5;
    classDef backbone fill:#FFF3E0,stroke:#E65100,stroke-width:2px,color:#BF360C,rx:5,ry:5;
    classDef pooling fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#4A148C,rx:5,ry:5;
    classDef fusion fill:#FCE4EC,stroke:#C2185B,stroke-width:2px,color:#880E4F,rx:5,ry:5;
    classDef dense fill:#FFFDE7,stroke:#FBC02D,stroke-width:2px,color:#F57F17,rx:5,ry:5;
    classDef scaling fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20,rx:5,ry:5;
    classDef output fill:#E0F2F1,stroke:#00695C,stroke-width:3px,color:#004D40,rx:10,ry:10;

    subgraph Inputs [输入层 Inputs]
        direction LR
        IMG_IN[图像输入<br>Image Input<br>(B, 3, 512, 512)]:::input
        SEX_IN[性别输入<br>Sex Input<br>(B, 1)]:::input
    end

    subgraph ImagePath [图像特征提取路径 Image Branch]
        IMG_IN --> BACKBONE[<b>ConvNeXt-Tiny Backbone</b><br>(预训练特征提取器)]:::backbone
        BACKBONE -- "(B, 768, H', W')" --> GEM[<b>GeM Pooling</b><br>(广义平均池化)]:::pooling
        GEM -- "(B, 768)" --> IMG_VEC(图像特征向量<br>Image Feature Vector):::dense
    end

    subgraph SexPath [性别编码路径 Sex Branch]
        SEX_IN --> SEX_ENC_1[Linear (1->16)<br>BN + ReLU]:::dense
        SEX_ENC_1 --> SEX_ENC_2[Linear (16->32)<br>BN + ReLU]:::dense
        SEX_ENC_2 -- "(B, 32)" --> SEX_VEC(性别特征向量<br>Sex Feature Vector):::dense
    end

    subgraph FusionHead [融合与回归头 Fusion & Regression Head]
        IMG_VEC --> CONCAT{特征拼接<br>Concatenation}:::fusion
        SEX_VEC --> CONCAT
        CONCAT -- "(B, 800)" --> HEAD_1[BN(800) + Dropout(0.5)]:::dense
        HEAD_1 --> HEAD_2[Linear (800->512)<br><b>Mish Activation</b>]:::dense
        HEAD_2 --> HEAD_3[BN(512) + Dropout(0.4)]:::dense
        HEAD_3 --> HEAD_4[Linear (512->64)<br><b>Mish Activation</b>]:::dense
        HEAD_4 --> HEAD_5[Linear (64->1)<br>Raw Output]:::dense
    end

    subgraph OutputLayer [输出层 Output Layer]
        HEAD_5 --> SIGMOID[<b>Range Scaling</b><br>Sigmoid * MaxAge(240)]:::scaling
        SIGMOID --> FINAL_OUT((最终预测骨龄<br>Final Bone Age<br>月份 Months)):::output
    end

    %% 样式调整
    linkStyle default stroke:#455A64,stroke-width:2px,fill:none;
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


