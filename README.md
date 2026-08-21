# dementia-mri-classification-densenet169

# Dementia Diagnosis from MRI using DenseNet169

Automated 4-class classification of Alzheimer's/dementia stage from brain MRI scans, 
using transfer learning on DenseNet169.

## Overview
- **Classes**: Non-Demented, Very Mild, Mild, Moderate Demented
- **Dataset**: Kaggle Alzheimer's MRI 4-Class Dataset
- **Preprocessing**: bilateral filtering (noise removal), histogram equalization 
  (contrast/edge enhancement), data augmentation for class balance
- **Model**: DenseNet169 (ImageNet weights), fine-tuned last conv block, 
  hyperparameters tuned via Optuna

## Results
| Metric | Score |
|---|---|
| Test Accuracy | 86.77% |
| Weighted F1-score | 0.87 |
| Validation Accuracy | 89.44% |

Per-class F1: Non-Demented 0.87, Very Mild Demented 0.99, Mild Demented 0.85, 
Moderate Demented 0.75

## Tech stack
Python, TensorFlow/Keras, OpenCV, scikit-learn, Optuna, pandas, matplotlib/seaborn

## Notebook
See `main_-_Densenet169.ipynb` for full pipeline: preprocessing → augmentation → 
model training → evaluation (confusion matrix, ROC curves, classification report).
