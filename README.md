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
