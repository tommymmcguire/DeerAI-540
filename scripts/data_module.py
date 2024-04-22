import os
import re
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class AgeDataset(Dataset):
    """Dataset class to load age-related image data."""
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

def get_transform():
    """Defines the transformation to apply to the images."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
