# Image Similarity Engine

## Overview
This project we attempt implement an Image Similarity Engine triplet loss-based approaches to evaluate image similarity. This aims to find similar images to the anchor image from database.

## Project Structure
```
ImageSimilarityEngine/
│── EvaluationModel/
│   │── best_model_classifier.pth        # Trained classifier model
│   │── best_model_triplet.pth           # Trained triplet model
│   │── evaluation_model.py              # Evaluation script
│
│── SimpleClassifier/
│   │── cifar_classifier.py              # CIFAR classifier implementation
│   │── cnn_network.py                   # CNN network for classification
│
│── Triplet/
│   │── datasets_s.py                    # Dataset handling for triplet loss
│   │── losses.py                         # Loss function implementations
│   │── main.py                           # Main script for training/testing
│   │── metrics.py                        # Metrics for evaluation
│   │── networks.py                       # Network architectures
│   │── tensorboard.ipynb                 # TensorBoard visualization
│   │── train.py                          # Training script
│   │── trainer.py                        # Model trainer logic
│   │── Triplet.ipynb                     # Notebook for training/testing
│   │── utils.py                          # Utility functions
```

## Installation
### Prerequisites
Ensure you have Python 3.8+ installed along with the following dependencies:
```bash
pip install torch torchvision numpy matplotlib tensorboard
```

## Usage
### Training
To train the classifier model:
```bash
python SimpleClassifier/cifar_classifier.py
```

To train the triplet model:
```bash
python Triplet/train.py
```

### Evaluation
To evaluate an image using the trained models:
```bash
python EvaluationModel/evaluation_model.py --model best_model_triplet.pth --image <image_path>
```

## Acknowledgments
This project uses PyTorch for deep learning and TensorBoard for visualization. The dataset used is assumed to be CIFAR or a similar dataset.

## Future Work
- Improve model performance with advanced architectures.
- Implement a user-friendly API for image similarity search.
- Explore additional datasets for better generalization.

