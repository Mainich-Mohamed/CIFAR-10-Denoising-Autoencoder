import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 32x32x3 -> 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32x32 -> 16x16x32

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 16x16x32 -> 16x16x64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16x64 -> 8x8x64

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 8x8x64 -> 8x8x128
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 8x8x128 -> 4x4x128
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 4x4x128 -> 8x8x64
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 8x8x64 -> 16x16x32
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 16x16x32 -> 32x32x3
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded