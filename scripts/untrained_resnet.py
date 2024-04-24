import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from math import sqrt
import matplotlib.pyplot as plt
from .data_module import AgeDataset, get_transform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_untrained_model():
    """Create an untrained ResNet model with a modified output layer."""
    model = models.resnet50(pretrained=False)  # Set pretrained=False for an untrained model
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model.to(device)

def evaluate_model(model, data_loader):
    """Evaluate the model using MAE and RMSE, and calculate accuracy for predictions within ±1 year."""
    actual_ages = []
    predicted_ages = []
    losses = []
    criterion = nn.MSELoss()
    predictions = []

    with torch.no_grad():
        for images, ages in tqdm(data_loader, desc="Evaluating Model"):
            images, ages = images.to(device), ages.to(device)
            outputs = model(images).squeeze()
            
            actual_ages.extend(ages.cpu().numpy())
            predicted_ages.extend(outputs.cpu().numpy())
            losses.append(criterion(outputs, ages.float()).item())

            # Calculate if each prediction is within ±1 year
            outputs_rounded = outputs.round().cpu().numpy()
            predictions.extend([abs(pred - act) <= 1 for pred, act in zip(outputs_rounded, ages.cpu().numpy())])

    # Metrics calculation
    mae = mean_absolute_error(actual_ages, predicted_ages)
    rmse = sqrt(mean_squared_error(actual_ages, predicted_ages))
    accuracy = accuracy_score([True] * len(predictions), predictions)
    avg_loss = np.mean(losses)

    print(f'Average Loss: {avg_loss:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(f'Accuracy for predictions within ±1 year: {accuracy:.4f}')

def evaluate_untrained_resnet():
    """Setup the dataset and model, then run evaluations."""
    transform = get_transform()
    test_dataset = AgeDataset('./datasets/test', transform=transform)  # Adjust path as necessary
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = create_untrained_model()
    evaluate_model(model, test_loader)


