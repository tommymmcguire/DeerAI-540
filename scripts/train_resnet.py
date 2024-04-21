import os
import re
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create a class to prepare the dataset
# This class will load the images and their corresponding ages
class AgeDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.filenames = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.filenames[idx])
        image = Image.open(img_name).convert('RGB')
        basename = os.path.basename(img_name)
        matches = re.findall(r'\d+', basename)
        if not matches:
            raise ValueError(f"No age found in filename: {img_name}")
        age = float(matches[-1])
        if self.transform:
            image = self.transform(image)
        return image, age

# Define the transformation to be applied to the images
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Create the ResNet model
def create_model():
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Train the ResNet model
def train_model(model, dataloaders, criterion, optimizer, num_epochs=30):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for images, ages in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Phase"):
                images = images.to(model.device)
                ages = ages.to(model.device).view(-1, 1).float()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images).view(-1, 1)
                    loss = criterion(outputs, ages)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}')

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # Load best model weights
    model.load_state_dict(best_model_wts)
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Lowest Validation Loss: {best_loss:.4f}')
    return best_model_wts

# Save the trained model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to '{path}'")

# This function will train the ResNet model and save the best weights
def train_resnet():
    transform = get_transform()
    datasets = {x: AgeDataset(f'../datasets/{x}', transform) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(datasets[x], batch_size=32, shuffle=x=='train', num_workers=4) for x in ['train', 'val']}

    model = create_model()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_model_wts = train_model(model, dataloaders, criterion, optimizer)
    
    # Save the best model weights
    model_save_path = '../model.pth'
    save_model(model, model_save_path)
