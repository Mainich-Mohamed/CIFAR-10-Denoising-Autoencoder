import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training hyperparameters
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 64
LEARNING_RATE = 0.0001
EPOCHS = 500
NOISE_FACTOR = 0.1
NUM_WORKERS = 2

# Visualization
NUM_IMAGES_TO_SHOW = 5