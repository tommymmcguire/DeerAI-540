import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
from math import sqrt
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the trained model
def load_model(model_path):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Define the AgeDataset class
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

# Evaluate the model for regression, which is what the model was trained for
def evaluate_regression(model, test_loader):
    criterion = nn.MSELoss()
    actuals, predictions = [], []
    test_loss = 0.0

    with torch.no_grad():
        for images, ages in tqdm(test_loader, desc="Testing Phase"):
            images = images.to(device)
            ages = ages.to(device).view(-1, 1).float()

            outputs = model(images).view(-1, 1)
            loss = criterion(outputs, ages)
            test_loss += loss.item() * images.size(0)

            actuals.extend(ages.cpu().numpy())
            predictions.extend(outputs.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    actuals = np.array(actuals).flatten()
    predictions = np.array(predictions).flatten()
    mae = mean_absolute_error(actuals, predictions)
    rmse = sqrt(mean_squared_error(actuals, predictions))
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')

# Map age to categories for classification
def map_age_to_category(age):
    if age <= 5:
        return '0-5'
    elif age <= 10:
        return '6-10'
    else:
        return 'older than 10'

# Evaluate the model for classification
def evaluate_classification(model, test_loader):
    categorized_actuals, categorized_predictions = [], []

    with torch.no_grad():
        for images, ages in tqdm(test_loader, desc="Testing Phase"):
            images = images.to(device)
            categorized_actual_ages = [map_age_to_category(age.item()) for age in ages.cpu()]

            outputs = model(images).view(-1).cpu().numpy()
            categorized_predicted_ages = [map_age_to_category(prediction) for prediction in outputs]

            categorized_actuals.extend(categorized_actual_ages)
            categorized_predictions.extend(categorized_predicted_ages)

    accuracy = accuracy_score(categorized_actuals, categorized_predictions)
    report = classification_report(categorized_actuals, categorized_predictions, target_names=['0-5', '6-10', 'older than 10'])

    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(report)

# This function evaluates the ResNet model
def evaluate_resnet():
    # Load the trained model
    model_path = '../model.pth'
    model = load_model(model_path)

    # Define the test dataset and DataLoader
    transform = get_transform()
    test_dataset = AgeDataset('../datasets/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Evaluate the model for regression
    print("Evaluating Regression Metrics:")
    evaluate_regression(model, test_loader)
    
    # Evaluate the model for classification
    print("\nEvaluating Classification Metrics:")
    evaluate_classification(model, test_loader)
