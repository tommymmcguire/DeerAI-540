import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torch.optim import Adam
from tqdm import tqdm
from .data_module import AgeDataset, get_transform 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_model():
    """Create and initialize the ResNet model for age prediction."""
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model.to(device)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=30):
    """Train the model and save the best performing model weights."""
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            for images, ages in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Phase"):
                images = images.to(device)
                ages = ages.to(device).view(-1, 1).float()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    loss = criterion(outputs, ages)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Lowest Validation Loss: {best_loss:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def save_model(model, path):
    """Save the trained model."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to '{path}'")

def train_resnet():
    """Main training function."""
    transform = get_transform()
    datasets = {x: AgeDataset(f'./datasets/{x}', transform) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(datasets[x], batch_size=32, shuffle=x == 'train', num_workers=4) for x in ['train', 'val']}
    
    model = create_model()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    model = train_model(model, dataloaders, criterion, optimizer)
    
    model_save_path = './model.pth'
    save_model(model, model_save_path)
