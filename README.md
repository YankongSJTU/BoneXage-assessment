# 🦴 BoneXage-Assessment

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A PyTorch implementation for precise pediatric Bone Age Assessment (BAA) based on hand radiographs.

This project utilizes a modern CNN architecture (ConvNeXt) combined with Generalized Mean (GeM) Pooling and specific optimizations for medical regression tasks. It achieves superior convergence and stability on the RSNA Pediatric Bone Age dataset compared to traditional baselines like ResNet or EfficientNet.

---

## 🚀 Key Features

* **SOTA Backbone**: Utilizes `ConvNeXt-Tiny` as the feature extractor, offering stronger feature representation for medical imaging analysis.
* **GeM Pooling**: Implements learnable Generalized Mean (GeM) Pooling instead of standard Average Pooling. This acts similarly to an attention mechanism, better capturing high-response features like crucial ossification centers.
* **Multi-Modal Fusion**: Integrates image features with gender encoding (male/female) to significantly improve prediction accuracy, as bone maturation rates differ by sex.
* **Numerical Stability**: Applies a `Sigmoid * Max_Age` (e.g., 240 months) scaling technique at the final output layer. This constraint eliminates physically impossible negative predictions and prevents gradient explosion, ensuring highly stable training dynamics.
* **AMP Support**: Implements Automatic Mixed Precision (AMP) for faster training speeds and reduced GPU memory usage.

## 📂 Project Structure

```text
BoneXage-Assessment/
├── data/                  # Dataset storage (must be created manually)
│   ├── boneage-training-dataset/
│   └── boneage-test-dataset/
├── checkpoints/           # Saved model weights
├── models.py              # Model architecture and configuration
├── train.py               # Training script
├── predict.py             # Inference script for batch prediction
├── requirements.txt       # Dependencies list
└── README.md              # Project documentation
