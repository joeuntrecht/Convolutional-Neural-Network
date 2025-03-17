# Convolutional Neural Network for Image Classification

## Overview
This project implements and trains a **LeNet-5 Convolutional Neural Network (CNN)** using **PyTorch** to classify images into 100 scene categories from the **MiniPlaces dataset** (120K images). The model architecture includes convolutional layers, max pooling, and fully connected layers. The project explores different hyperparameter settings to optimize model performance.

## Features
- **Custom LeNet-5 Architecture:**
  - 2 Convolutional layers
  - Max pooling layers
  - Fully connected layers
  - ReLU activations
- **Hyperparameter Optimization:**
  - Batch size tuning
  - Learning rate adjustments
  - Number of training epochs
- **Performance Evaluation:**
  - Tracks validation accuracy
  - Logs training metrics for analysis

## File Structure
```
├── CNN/
│   ├── lenet.py            # Defines the LeNet-5 architecture
│   ├── dataloader.py       # Handles data loading and preprocessing
│   ├── train_miniplaces.py # Training script
│   ├── eval_miniplaces.py  # Evaluation script
│   ├── checkpoint.pth.tar  # Model checkpoint file
│   ├── model_best.pth.tar  # Best model checkpoint
│   ├── results.txt         # Training results (accuracy/loss)
```

## Setup & Installation
### Prerequisites
- Python 3.9
- PyTorch
- TorchVision
- tqdm

### Install Dependencies
```bash
pip install torch torchvision tqdm
```

## Training the Model
To train the model, run:
```bash
python train_miniplaces.py --epochs 10 --lr 0.001 --batch-size 32
```
- **`--epochs`**: Number of training epochs
- **`--lr`**: Learning rate
- **`--batch-size`**: Number of images per mini-batch

## Evaluating the Model
To evaluate the trained model:
```bash
python eval_miniplaces.py --load ./outputs/model_best.pth.tar
```


## Future Improvements
- Experiment with **more complex architectures** like ResNet or EfficientNet.
- Implement **data augmentation techniques** to improve generalization.
- Fine-tune on pre-trained models for better performance.
