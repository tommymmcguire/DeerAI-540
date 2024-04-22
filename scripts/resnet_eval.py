import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
from math import sqrt
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_module import AgeDataset, get_transform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def is_accurate_prediction(predicted, actual):
    """Returns True if the prediction is within ±1 year of the actual age, otherwise False."""
    return abs(predicted - actual) <= 1

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

def evaluate_accuracy(model, test_loader):
    predictions = []
    with torch.no_grad():
        for images, ages in tqdm(test_loader, desc="Accuracy Testing"):
            images = images.to(device)
            ages = ages.cpu().numpy()  # Actual ages

            outputs = model(images).view(-1).cpu().numpy()  # Predicted ages

            # Compare each predicted age to the actual age and classify as correct or incorrect
            predictions.extend([is_accurate_prediction(pred, act) for pred, act in zip(outputs, ages)])

    # Calculate Accuracy
    accuracy = accuracy_score([True] * len(predictions), predictions)  # Using True as all attempts are to be accurate
    print(f'Accuracy for predictions within ±1 year: {accuracy:.4f}')

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
    
    print("\nEvaluating Accuracy for close predictions:")
    evaluate_accuracy(model, test_loader)
