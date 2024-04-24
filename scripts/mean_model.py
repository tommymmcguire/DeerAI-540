import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from .data_module import AgeDataset, get_transform

# Function to calculate the mean age from a DataLoader
def calculate_mean_age(loader):
    total_age = 0.0
    count = 0
    for _, ages in tqdm(loader, desc="Calculating mean age"):
        total_age += ages.sum().item()
        count += len(ages)
    return total_age / count

# Function to check if predictions are within a certain tolerance
def is_within_tolerance(predicted, actual, tolerance):
    return abs(predicted - actual) <= tolerance

# Function to map age to categorical labels
def map_age_to_category(age):
    if age <= 5:
        return '0-5'
    elif age <= 10:
        return '6-10'
    else:
        return 'older than 10'

# Function to predict mean age and evaluate using the DataLoader
def predict_and_evaluate_mean_age(loader, mean_age):
    actual_ages = []
    predicted_ages = []
    for _, ages in tqdm(loader, desc="Predicting and evaluating mean age"):
        batch_size = ages.size(0)
        actual_ages.extend(ages.numpy())
        predicted_ages.extend([mean_age] * batch_size)

    mse = mean_squared_error(actual_ages, predicted_ages)
    mae = mean_absolute_error(actual_ages, predicted_ages)
    rmse = np.sqrt(mse)

    tolerance_1_year = [is_within_tolerance(pred, act, 1) for pred, act in zip(predicted_ages, actual_ages)]
    tolerance_2_years = [is_within_tolerance(pred, act, 2) for pred, act in zip(predicted_ages, actual_ages)]
    accuracy_1_year = accuracy_score([True] * len(tolerance_1_year), tolerance_1_year)
    accuracy_2_years = accuracy_score([True] * len(tolerance_2_years), tolerance_2_years)

    categorized_actuals = [map_age_to_category(age) for age in actual_ages]
    categorized_predictions = [map_age_to_category(pred) for pred in predicted_ages]
    accuracy_categorized = accuracy_score(categorized_actuals, categorized_predictions)
    report = classification_report(categorized_actuals, categorized_predictions, target_names=['0-5', '6-10', 'older than 10'])

    return mse, mae, rmse, accuracy_1_year, accuracy_2_years, accuracy_categorized, report

# Main function to run evaluations
def run_mean_model():
    transform = get_transform()
    test_dataset = AgeDataset('./datasets/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    mean_age = calculate_mean_age(test_loader)
    print(f"Calculated Mean Age: {mean_age:.2f}")

    metrics = predict_and_evaluate_mean_age(test_loader, mean_age)
    print(f"MSE: {metrics[0]:.4f}, MAE: {metrics[1]:.4f}, RMSE: {metrics[2]:.4f}")
    print(f"Accuracy within ±1 year: {metrics[3]:.4f}")
    print(f"Accuracy within ±2 years: {metrics[4]:.4f}")
    print(f"Categorical Accuracy: {metrics[5]:.4f}")
    print('Classification Report:')
    print(metrics[6])

