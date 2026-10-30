# Model Checkpoints Directory

Pretrained model weights for the CIFAR-10 ResNet-18 model are stored in this directory as `resnet18_cifar10.pt`.

### Training the Model
To train the ResNet-18 model on CIFAR-10 from scratch:
```bash
# Standard training (3 epochs, ~90 seconds on Apple Silicon MPS / GPU):
python train.py --epochs 3 --batch-size 128 --train-subset 8000 --val-subset 1500

# Quick-start benchmark (1 epoch on 2,000 samples):
python train.py --quick
```

Trained weights will automatically be saved to `checkpoints/resnet18_cifar10.pt`, and training history to `results/training_history.json`.
