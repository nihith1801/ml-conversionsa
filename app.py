import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Paths to  models and labels
model_paths = {
    "Breast Cancer Ultrasound Classification Model": ("my_model/mri_model.h5", "my_model/mri_labels.txt"),
    "Bone Fracture Classification Model": ("my_model/Bone_Fracture.h5", "my_model/bone_fracture_labels.txt"),
    "Brain Tumor Classification Model": ("my_model/brain_tumor_model.h5", "my_model/brain_tumor_labels.txt"),
}

# Function to load a Keras model
def load_keras_model(model_path):
    return load_model(model_path, compile=False)

# Function to load labels from a file
def load_labels(labels_path):
    with open(labels_path, "r") as file:
        return [line.strip() for line in file.readlines()]

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox("Hey there!\nChoose a model", list(model_paths.keys()))

# Load the selected model and corresponding labels
model_path, labels_path = model_paths[model_choice]
model = load_keras_model(model_path)
class_names = load_labels(labels_path)

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Function to preprocess the image
def preprocess_image(image):
    size = (224, 224)  # Resize to the model's expected input size
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return np.expand_dims(normalized_image_array, axis=0)

# Display the image and make predictions
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    data = preprocess_image(image)

    # Predict the class
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display the prediction
    st.write(f"**Prediction:** {class_name}")
    st.write(f"**Confidence Score:** {confidence_score:.2f}")
