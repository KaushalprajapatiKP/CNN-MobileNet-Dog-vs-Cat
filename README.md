## Dog vs Cat Classification using MobileNet

This project implements a deep learning model to classify images of dogs and cats using the MobileNet architecture. It leverages the Kaggle API for dataset retrieval and includes training, evaluation, and deployment steps.

### Features

**Efficient Model** : Utilizes MobileNet for fast and lightweight image classification.

**Dataset Integration**: Automatically downloads and extracts the dataset from Kaggle.

**Training Pipeline**: Includes data preprocessing, model training, and evaluation.

**High Accuracy**: Aims for high classification accuracy on unseen data.

Setup Instructions

Prerequisites

Ensure you have the following installed:

Python 3.7+

TensorFlow

Kaggle API

Installation

Clone the repository:

git clone https://github.com/yourusername/dog-vs-cat-mobilenet.git
cd dog-vs-cat-mobilenet

Install dependencies:

pip install -r requirements.txt

Configure Kaggle API:

Place your kaggle.json file in the project directory.

Run the following commands to set up:

mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

Download the dataset:

kaggle datasets download tongpython/cat-and-dog

Run .ipynb file for running the model

Model Architecture

The project uses MobileNet, a lightweight deep learning architecture designed for resource-constrained environments. The model is fine-tuned on the dog vs. cat dataset for optimal performance.

Dataset

The dataset is sourced from Kaggle and contains labeled images of cats and dogs. It is preprocessed into training and validation splits for model development.

Results

Training Accuracy: Achieved high accuracy on training data.

Validation Accuracy: Robust performance on unseen data.

Detailed metrics and visualizations can be found in the accompanying Jupyter notebooks.


Future Improvements

Implementing additional augmentations for data diversity.

Exploring other lightweight architectures.

Deploying the model as a web application.

Acknowledgments

Kaggle for providing the dataset.

TensorFlow for the MobileNet implementation.

