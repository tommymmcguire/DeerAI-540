import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import os

# Load your model (ensure the function is defined correctly)
def load_model():
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()


def predict_age(image, bbox):
    # Crop the image to the bounding box [x_min, y_min, width, height]
    cropped_image = image.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
    # Transform the image and add batch dimension
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(cropped_image).unsqueeze(0)
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        predicted_age = output.item()
    return predicted_age

st.title('Deer Age Prediction')

uploaded_file = st.file_uploader("Upload an image of a deer", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Create a canvas and let the user draw the bounding box
    bbox = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Use a transparent fill color
        stroke_width=5,
        stroke_color="#e00",
        background_image=Image.open(uploaded_file).convert("RGBA"),
        update_streamlit=True,
        width=image.width,
        height=image.height,
        drawing_mode="rect",
        key="canvas",
    )

    if bbox.json_data is not None:
        shapes = bbox.json_data["objects"]
        if shapes:
            # Assumes only one rectangle is drawn
            rect = shapes[0]
            bounds = rect["left"], rect["top"], rect["width"], rect["height"]
            predicted_age = predict_age(image, bounds)
            st.write(f"The predicted age of the deer is: {predicted_age:.2f} years")
