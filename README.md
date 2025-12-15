# CIFAR-10 Denoising Autoencoder

A PyTorch implementation of a convolutional denoising autoencoder designed to remove Gaussian noise from CIFAR-10 images. This project demonstrates deep learning techniques for image denoising, achieving high PSNR and SSIM scores.

![IoT System Architecture](https://www.mdpi.com/IoT/IoT-04-00016/article_deploy/html/images/IoT-04-00016-g001.png)

## Features ✅
- ***Convolutional Architecture***: Uses convolutional and transposed convolutional layers for efficient feature extraction and reconstruction
- ***Noise Removal:*** Effectively removes Gaussian noise (configurable std) from 32x32 RGB images.
- ***Metrics Evaluation:*** Computes PSNR and SSIM for quantitative assessment.
- ***Visualization:*** Generates side-by-side plots of original, noisy, and denoised images.
- ***Configurable Training:*** Adjustable epochs, learning rate, noise factor, and batch sizes.
- ***GPU Acceleration***: Automatic CUDA support for faster training and inference


## Performance Metrics 📊 

- **Average PSNR**: 24.5 dB
- **Average SSIM**: 0.815
- **Dataset**: CIFAR-10 (60,000 32x32 color images)
- **Noise Type**: Additive Gaussian Noise (σ=0.1)

## Denoising Performance 📝
The model demonstrates effective noise removal while preserving important image features. Below is a sample of the denoising results:

<img width="1442" height="2490" alt="CIFAR-10-Autoencoder-Results" src="https://github.com/user-attachments/assets/f9beedd6-3d12-4c32-928d-6ea8971a0d49" />

## Quantitative Results:

- PSNR: 24.5 dB (Higher is better)
- SSIM: 0.815 (Closer to 1.0 is better)

##  Future Improvements 🛠️
- ***Architecture Enhancements:*** Add skip connections (U-Net style) and residual blocks
- ***Advanced Loss Functions:*** Incorporate perceptual loss or adversarial training
- ***Noise Adaptation:*** Train with varying noise levels for robustness
- ***Batch Normalization:*** Add BatchNorm layers for faster convergence
