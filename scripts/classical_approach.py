import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import re

class ClassicalApproach:
    def __init__(self):
        self.train_dir = './datasets/train'
        self.val_dir = './datasets/val'
        self.test_dir = './datasets/test'
        self.model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

    def resize_image(self, image, target_height=500):
        """Resize images while keeping aspect ratio."""
        h, w = image.shape[:2]
        scaling_factor = target_height / h
        return cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    def extract_shape_features(self, image_path):
        """Extract basic shape features related to the aspect ratio of the deer's silhouette."""
        image = cv2.imread(image_path)
        image = self.resize_image(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h
            return [aspect_ratio]
        return [0]  # No contour found

    def process_image(self, data):
        filename, directory = data
        try:
            image_path = os.path.join(directory, filename)
            if filename.lower().endswith(('.jpg', '.jpeg')):
                aspect_ratio = self.extract_shape_features(image_path)[0]
                matches = re.findall(r'\d+', filename)
                if not matches:
                    raise ValueError(f"No age found in filename: {image_path}")
                age = float(matches[-1])
                return aspect_ratio, age
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
        return 0, 0  # Default values in case of an error

    def prepare_dataset_parallel(self, directory):
        tasks = [(filename, directory) for filename in os.listdir(directory) if filename.lower().endswith(('.jpg', '.jpeg'))]
        features, ages = [], []
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.process_image, task): task for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing images in {directory}"):
                aspect_ratio, age = future.result()
                features.append([aspect_ratio])
                ages.append(age)
        return np.array(features).reshape(-1, 1), np.array(ages)

    def train_model(self):
        X_train, y_train = self.prepare_dataset_parallel(self.train_dir)
        self.model.fit(X_train, y_train)
        return self.evaluate_model(X_train, y_train, "Training")

    def evaluate_model(self, features, ages, phase="Validation"):
        predictions = self.model.predict(features)
        mse = mean_squared_error(ages, predictions)
        mae = mean_absolute_error(ages, predictions)
        rmse = np.sqrt(mse)
        print(f'{phase} MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')

    def run(self):
        print("Training model...")
        self.train_model()
        X_val, y_val = self.prepare_dataset_parallel(self.val_dir)
        print("Validating model...")
        self.evaluate_model(X_val, y_val)
        X_test, y_test = self.prepare_dataset_parallel(self.test_dir)
        print("Testing model...")
        self.evaluate_model(X_test, y_test, "Test")
