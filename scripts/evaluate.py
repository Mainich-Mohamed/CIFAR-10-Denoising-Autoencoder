import torch
import matplotlib.pyplot as plt
import numpy as np
from models.autoencoder import DenoisingAutoencoder
from utils.data_utils import get_data_loaders, add_gaussian_noise
from utils.metrics import calculate_metrics
import config

def evaluate_and_visualize(model_path='final_model.pth', num_images=config.NUM_IMAGES_TO_SHOW):
    # Get test loader
    _, test_loader = get_data_loaders(
        batch_size_train=config.BATCH_SIZE_TRAIN,
        batch_size_test=config.BATCH_SIZE_TEST,
        num_workers=config.NUM_WORKERS
    )
    
    # Load model
    model = DenoisingAutoencoder().to(config.DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Collect results
    all_clean = []
    all_noisy = []
    all_recon = []
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        collected = 0
        for data, _ in test_loader:
            if collected >= num_images:
                break
                
            data = data.to(config.DEVICE)
            noisy_data = add_gaussian_noise(data, config.NOISE_FACTOR)
            reconstructed = model(noisy_data)
            
            num_to_take = min(num_images - collected, data.size(0))
            for i in range(num_to_take):
                clean_img = data[i].cpu().numpy().transpose(1, 2, 0)
                noisy_img = noisy_data[i].cpu().numpy().transpose(1, 2, 0)
                recon_img = reconstructed[i].cpu().numpy().transpose(1, 2, 0)
                
                clean_img = np.clip(clean_img, 0, 1)
                noisy_img = np.clip(noisy_img, 0, 1)
                recon_img = np.clip(recon_img, 0, 1)
                
                all_clean.append(clean_img)
                all_noisy.append(noisy_img)
                all_recon.append(recon_img)
                
                psnr_val, ssim_val = calculate_metrics(clean_img, recon_img)
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)
                
                collected += 1
    
    # Plot results
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    for i in range(num_images):
        axes[i, 0].imshow(all_clean[i])
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(all_noisy[i])
        axes[i, 1].set_title('Noisy')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(all_recon[i])
        axes[i, 2].set_title(f'Denoised\nPSNR: {psnr_values[i]:.2f}, SSIM: {ssim_values[i]:.3f}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('denoising_results.png')
    plt.show()
    
    # Print average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    print(f'Average PSNR: {avg_psnr:.2f}')
    print(f'Average SSIM: {avg_ssim:.3f}')
    
    return avg_psnr, avg_ssim

if __name__ == "__main__":
    evaluate_and_visualize()