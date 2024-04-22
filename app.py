import streamlit as st
from PIL import Image, ImageDraw
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
def predict_age(image, bbox):
    # Crop the image to the bounding box
    cropped_image = image.crop(bbox)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(cropped_image).unsqueeze(0)
    # Predict the age using the model
    with torch.no_grad():
        output = model(input_tensor)
        predicted_age = output.item()
    return predicted_age

# Function to draw bounding box on image
def draw_bbox_on_image(image, bbox):
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline="red", width=3)
    return image

# Main app interface
st.title('Deer Age Prediction App')
st.write("Upload an image of a deer and create a bounding box by selecting the coordinates for the top-left and bottom-right corners.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True, caption='Uploaded Image')

    if 'points' not in st.session_state:
        st.session_state['points'] = []

    if st.button("Clear Points"):
        st.session_state['points'] = []

    cols = st.columns(2)
    with cols[0]:
        x = st.number_input('X coordinate', min_value=0, max_value=image.width, step=1)
    with cols[1]:
        y = st.number_input('Y coordinate', min_value=0, max_value=image.height, step=1)

    if st.button('Add Point'):
        if len(st.session_state['points']) < 2:
            st.session_state['points'].append((x, y))
        else:
            st.warning('Only two points are needed, please clear points if you want to reselect.')

    if len(st.session_state['points']) == 2:
        bbox = [st.session_state['points'][0][0], st.session_state['points'][0][1], 
                st.session_state['points'][1][0], st.session_state['points'][1][1]]
        display_image = draw_bbox_on_image(image.copy(), bbox)
        st.image(display_image, use_column_width=True, caption='Image with Bounding Box')

        if st.button('Predict Age'):
            predicted_age = predict_age(image, bbox)
            st.success(f"Predicted age for Deer: {predicted_age:.2f} years")
