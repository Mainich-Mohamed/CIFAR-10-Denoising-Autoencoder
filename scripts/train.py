import torch
import torch.optim as optim
from models.autoencoder import DenoisingAutoencoder
from utils.data_utils import get_data_loaders, add_gaussian_noise
import config
import matplotlib.pyplot as plt

def train():
    # Get data loaders
    train_loader, _ = get_data_loaders(
        batch_size_train=config.BATCH_SIZE_TRAIN,
        batch_size_test=config.BATCH_SIZE_TEST,
        num_workers=config.NUM_WORKERS
    )
    
    # Initialize model
    model = DenoisingAutoencoder().to(config.DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    train_losses = []
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(config.DEVICE)
            noisy_data = add_gaussian_noise(data, config.NOISE_FACTOR)
            
            optimizer.zero_grad()
            outputs = model(noisy_data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{config.EPOCHS}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{config.EPOCHS}] completed. Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.close()
    
    return model

if __name__ == "__main__":
    trained_model = train()