import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn

# Set page configuration
st.set_page_config(layout="wide", page_title="Deer Age Prediction App")

# Load the model
def load_model():
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Function to predict age
def predict_age(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    # Predict the age using the model
    with torch.no_grad():
        output = model(input_tensor)
        predicted_age = output.item()
    return predicted_age

# Main app interface
st.title('Deer Age Prediction App')
st.write("""
         Please upload an image of the deer you would like to age. \n 
         To enhance the accuracy of the prediction, make sure that the deer is prominently displayed and isolated from its surroundings. \n
         Clear and unobstructed views of the deer yield the most reliable results.
         """)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Resize the image for display
    base_width = 500
    w_percent = (base_width / float(image.size[0]))
    h_size = int((float(image.size[1]) * float(w_percent)))
    display_image = image.resize((base_width, h_size), Image.Resampling.LANCZOS) 

    st.image(display_image, use_column_width=False, caption='Uploaded Image')

    if st.button('Predict Age'):
        predicted_age = predict_age(image)
        st.success(f"Predicted age for the deer: {predicted_age:.2f} years")
