# import streamlit as st
# from streamlit_drawable_canvas import st_canvas
# from PIL import Image, ImageDraw, ImageFont
# import torch
# from torchvision import models, transforms
# import torch.nn as nn

# # Set page configuration
# st.set_page_config(layout="wide", page_title="Deer Age Prediction App")

# # Load the model
# def load_model():
#     model = models.resnet50(pretrained=True)
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, 1)
#     model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
#     model.eval()
#     return model

# model = load_model()

# # Function to predict age
# def predict_age(image, bbox, display_scale):
#     try:
#         # Adjust bbox coordinates for original image dimensions
#         adjusted_bbox = [bbox[0] / display_scale, bbox[1] / display_scale, bbox[2] / display_scale, bbox[3] / display_scale]
#         # Crop the image to the bounding box
#         cropped_image = image.crop((adjusted_bbox[0], adjusted_bbox[1], adjusted_bbox[0]+adjusted_bbox[2], adjusted_bbox[1]+adjusted_bbox[3]))
#         preprocess = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#         input_tensor = preprocess(cropped_image).unsqueeze(0)
#         # Predict the age using the model
#         with torch.no_grad():
#             output = model(input_tensor)
#             predicted_age = output.item()
#         return predicted_age
#     except Exception as e:
#         st.error(f"Error in prediction: {str(e)}")
#         return None

# # Function to draw labels on an image
# def draw_labels_on_image(image, bboxes, ages):
#     draw = ImageDraw.Draw(image)
#     # Load a TrueType font (.ttf). Adjust the path to where the font file is stored on your system.
#     # For example, you might download and use a font like Arial.ttf. Here, we assume it's in the same directory.
#     try:
#         font = ImageFont.truetype("arial.ttf", 16)  # You can change "16" to another size as needed
#     except IOError:
#         # If the TTF file isn't found, fall back to the default font (not adjustable in size).
#         font = ImageFont.load_default()

#     for bbox, age in zip(bboxes, ages):
#         (x, y, w, h) = bbox
#         label = f"Deer: {age:.2f} years"
#         # Adjusting the position to place the text above the box by a fixed amount.
#         text_y_position = y - 20  # Change "-20" to adjust the vertical position as necessary.
        
#         # Draw bounding box
#         draw.rectangle(((x, y), (x+w, y+h)), outline="red", width=3)
#         # Draw text
#         draw.text((x, text_y_position), label, fill="red", font=font)

#     return image

# # Main app interface
# st.title('Deer Age Prediction App')
# uploaded_file = st.file_uploader("Upload an image of a deer", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")

#     # Resize image for display if it's too large
#     max_display_size = 800
#     display_scale = 1  # Default scale is 1 (no scaling)
#     if image.width > max_display_size or image.height > max_display_size:
#         display_scale = min(max_display_size / image.width, max_display_size / image.height)
#         display_image = image.resize((int(image.width * display_scale), int(image.height * display_scale)), Image.Resampling.LANCZOS)
#     else:
#         display_image = image.copy()  # Create a copy for drawing

#     canvas_width = display_image.width
#     canvas_height = display_image.height

#     bbox = st_canvas(
#         fill_color="rgba(255, 165, 0, 0.3)",
#         stroke_width=5,
#         stroke_color="#e00",
#         background_image=display_image,
#         update_streamlit=True,
#         width=canvas_width,
#         height=canvas_height,
#         drawing_mode="rect",
#         key="canvas",
#     )

#     if st.button('Predict Age for All Bounding Boxes'):
#         bboxes = []
#         ages = []
#         if bbox.json_data and bbox.json_data["objects"]:
#             for index, shape in enumerate(bbox.json_data["objects"], start=1):
#                 rect = [shape["left"], shape["top"], shape["width"], shape["height"]]
#                 predicted_age = predict_age(image, rect, display_scale)
#                 if predicted_age is not None:
#                     bboxes.append(rect)
#                     ages.append(predicted_age)
#                     st.write(f"Predicted age for Deer {index}: {predicted_age:.2f} years")

#             annotated_image = draw_labels_on_image(display_image, bboxes, ages)
#             st.write("### Processed Image with Age Predictions:")
#             st.image(annotated_image, caption='Annotated Image with Predicted Ages', use_column_width=True)

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import models, transforms
import torch.nn as nn
import os

# Set page configuration
st.set_page_config(layout="wide", page_title="Deer Age Prediction App")

# Load the model
def load_model():
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    # Ensure the model path is handled correctly across environments
    model_path = os.path.join('model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    else:
        st.error("Failed to load model. Check the model path.")
        raise FileNotFoundError("Model file not found")
    return model

model = load_model()

# Function to predict age
def predict_age(image, bbox, display_scale):
    try:
        adjusted_bbox = [bbox[0] / display_scale, bbox[1] / display_scale, bbox[2] / display_scale, bbox[3] / display_scale]
        cropped_image = image.crop((adjusted_bbox[0], adjusted_bbox[1], adjusted_bbox[0]+adjusted_bbox[2], adjusted_bbox[1]+adjusted_bbox[3]))
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(cropped_image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            predicted_age = output.item()
        return predicted_age
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

# Function to draw labels on an image
def draw_labels_on_image(image, bboxes, ages):
    draw = ImageDraw.Draw(image)
    # Use a more robust font handling approach
    font_path = "arial.ttf"
    try:
        font = ImageFont.truetype(font_path, 16)
    except IOError:
        font = ImageFont.load_default()

    for bbox, age in zip(bboxes, ages):
        (x, y, w, h) = bbox
        label = f"Deer: {age:.2f} years"
        text_y_position = y - 20
        
        draw.rectangle(((x, y), (x+w, y+h)), outline="red", width=3)
        draw.text((x, text_y_position), label, fill="red", font=font)

    return image

# Main app interface
st.title('Deer Age Prediction App')
uploaded_file = st.file_uploader("Upload an image of a deer", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    max_display_size = 800
    display_scale = 1
    if image.width > max_display_size or image.height > max_display_size:
        display_scale = min(max_display_size / image.width, max_display_size / image.height)
        display_image = image.resize((int(image.width * display_scale), int(image.height * display_scale)), Image.Resampling.LANCZOS)
    else:
        display_image = image.copy()

    canvas_width = display_image.width
    canvas_height = display_image.height

    bbox = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=5,
        stroke_color="#e00",
        background_image=display_image,
        update_streamlit=True,
        width=canvas_width,
        height=canvas_height,
        drawing_mode="rect",
        key="canvas",
        )

    if st.button('Predict Age for All Bounding Boxes'):
        bboxes = []
        ages = []
        if bbox.json_data and bbox.json_data["objects"]:
            for index, shape in enumerate(bbox.json_data["objects"], start=1):
                rect = [shape["left"], shape["top"], shape["width"], shape["height"]]
                predicted_age = predict_age(image, rect, display_scale)
                if predicted_age is not None:
                    bboxes.append(rect)
                    ages.append(predicted_age)
                    st.write(f"Predicted age for Deer {index}: {predicted_age:.2f} years")

            annotated_image = draw_labels_on_image(display_image, bboxes, ages)
            st.write("### Processed Image with Age Predictions:")
            st.image(annotated_image, caption='Annotated Image with Predicted Ages', use_column_width=True)
