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


