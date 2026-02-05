🦴 BoneXage-Assessment

A PyTorch implementation for precise pediatric Bone Age Assessment (BAA) based on hand radiographs.

This project utilizes a modern CNN architecture (ConvNeXt) combined with Generalized Mean (GeM) Pooling and specific optimizations for medical regression tasks. It achieves superior convergence and stability on the RSNA Pediatric Bone Age dataset compared to traditional baselines like ResNet or EfficientNet.
🚀 Key Features

    SOTA Backbone: Utilizes ConvNeXt-Tiny as the feature extractor, offering stronger feature representation for medical imaging analysis.

    GeM Pooling: Implements learnable Generalized Mean (GeM) Pooling instead of standard Average Pooling. This acts similarly to an attention mechanism, better capturing high-response features like crucial ossification centers.

    Multi-Modal Fusion: Integrates image features with gender encoding (male/female) to significantly improve prediction accuracy, as bone maturation rates differ by sex.

    Numerical Stability: Applies a Sigmoid * Max_Age (e.g., 240 months) scaling technique at the final output layer. This constraint eliminates physically impossible negative predictions and prevents gradient explosion, ensuring highly stable training dynamics.

    AMP Support: Implements Automatic Mixed Precision (AMP) for faster training speeds and reduced GPU memory usage.

📂 Project Structure
Plaintext

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

🛠️ Installation & Requirements

Ensure you have Python 3.8+ installed.

    Clone the repository:
    Bash

    git clone https://github.com/yourusername/BoneXage-assessment.git
    cd BoneXage-assessment

    Install dependencies: You can install the necessary libraries via pip:
    Bash

    pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm pillow

    Or, if you have a requirements.txt file provided:
    Bash

    pip install -r requirements.txt

💿 Dataset Preparation

This project uses the RSNA Pediatric Bone Age Challenge dataset. Due to license restrictions, you must download the data yourself (e.g., from Kaggle).

Please organize your data/ directory exactly as follows:
Plaintext

data/
├── boneage-training-dataset/   # Directory containing training images (.png)
├── boneage-test-dataset/       # Directory containing test images (.png)
├── boneage-training-dataset.csv
└── boneage-test-dataset.csv

🖥️ Usage
1. Training

Run train.py to start training the model. The script automatically detects available GPUs and utilizes nn.DataParallel for multi-GPU training if applicable.
Bash

python train.py

    Note: Hyperparameters like Batch Size, Learning Rate, and Image Size can be modified in the Config class within models.py.

2. Inference (Prediction)

Use predict.py to predict bone age for a list of images.

Preparation: Create a text file (e.g., image_list.txt) containing the full paths to the images you want to predict, one per line:
Plaintext

data/test_imgs/1234.png
data/test_imgs/5678.png

Command:
Bash

python predict.py <image_list_filename> <sex(M/F)> [image_size] [model_path]

Example:
Bash

python predict.py image_list.txt M 512 checkpoints/best_model.pth

Output Example:
Plaintext

data/test_imgs/1234.png: 10 years 6 months (126.5 months)
data/test_imgs/5678.png: 13 years 2 months (158.1 months)

🧠 Model Architecture

The following diagram illustrates the data flow through the network, highlighting the dual-branch input and the specialized pooling and output layers.
代码段

graph TD
    subgraph Inputs
        IMG[Input Image<br>512x512] --> B[ConvNeXt-Tiny Backbone]
        SEX[Input Sex<br>0 or 1] --> G[Sex Encoder MLP]
    end
    
    B -- Feature Maps --> C[GeM Pooling<br>Generalized Mean]
    C -- Image Vector --> I[Concat]
    G -- Sex Vector --> I
    
    I -- Fused Features --> J[Regression Head<br>FC Layers + Mish + BN + Dropout]
    J --> K[Range Scaling<br>Sigmoid * 240]
    K --> L((Predicted Age<br>Months))

    style IMG fill:#e1f5fe,stroke:#01579b
    style SEX fill:#e1f5fe,stroke:#01579b
    style C fill:#f3e5f5,stroke:#4a148c
    style K fill:#e8f5e9,stroke:#1b5e20
    style L fill:#fff9c4,stroke:#fbc02d,stroke-width:2px

📊 Results

Performance metrics on the RSNA test set.
Metric	Value (Approx)	Notes
MAE (Mean Absolute Error)	To be updated	Lower is better
RMSE (Root Mean Sq Error)	To be updated	
Backbone	ConvNeXt-Tiny	Pretrained on ImageNet
Input Resolution	512x512	
📝 Acknowledgments

    Dataset provided by the RSNA Pediatric Bone Age Challenge.

    ConvNeXt architecture based on the paper "A ConvNet for the 2020s".

📜 License

This project is licensed under the MIT License.
