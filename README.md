# Image Similarity Engine

## Overview
The Image Similarity Engine is a deep learning-based system that identifies similar images from a dataset given an input image. This project leverages Siamese Networks with Convolutional Neural Networks (CNNs) to compute image similarity.

## Features
- Uses Siamese Networks to compare images
- Supports feature extraction and similarity measurement
- Provides an interface to input images and retrieve similar images from a dataset

## Installation

1. Clone the repository or download the project files.
2. Install the required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare the Dataset
Place images to be compared inside the `dataset/` directory.

### 2. Train the Model
Run the following command to train the Siamese Network on your dataset:
   ```bash
   python train.py
   ```
This will generate trained model weights stored in `models/`.

### 3. Perform Image Similarity Search
To find similar images for a given input image, run:
   ```bash
   python search.py --input path/to/input_image.jpg
   ```
This will return the most similar images from the dataset.

## File Structure
```
ImageSimilarityEngine/
│── dataset/                # Folder containing dataset images
│── models/                 # Saved model weights
│── notebooks/              # Jupyter notebooks for experiments
│── src/                    # Source code files
│   │── siamese_model.py    # Model definition
│   │── train.py            # Training script
│   │── search.py           # Similarity search script
│── requirements.txt        # Required dependencies
│── README.md               # Project documentation
```

## Dependencies
Ensure you have the following dependencies installed:
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

## Contributors
- Prathamesh Pawar

## License
This project is licensed under the MIT License.

