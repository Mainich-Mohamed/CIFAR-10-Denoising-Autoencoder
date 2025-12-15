import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size_train=128, batch_size_test=64, num_workers=2):
    # Transform the PIL into a Tensor with a [0, 1] normalisation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Import and normalise the CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        "./data", train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        "./data", train=False, transform=transform, download=True
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_train, 
        shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size_test, 
        shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader

def add_gaussian_noise(images, noise_factor=0.1):
    # Add noise to the images
    noise = torch.randn_like(images) * noise_factor
    noisy_images = images + noise

    # Clip images to be between [0, 1]
    return torch.clamp(noisy_images, 0., 1.)