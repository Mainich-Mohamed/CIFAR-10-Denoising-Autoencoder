import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(clean_img, recon_img):
    psnr_val = psnr(clean_img, recon_img, data_range=1.0)
    ssim_val = ssim(clean_img, recon_img, data_range=1.0, channel_axis=-1)
    return psnr_val, ssim_val