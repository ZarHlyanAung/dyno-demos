import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

# Streamlit app title and description
st.title("Image Classification App")
st.write("Upload an image to classify using the pre-trained model.")

# URL of the model file on GitHub (replace with your actual URL)
model_url = "https://github.com/ZarHlyanAung/dyno-demos/raw/main/dyno1.h5"


# Path to store the downloaded model
model_path = "dyno1.h5"

# Function to download the model
def download_model(url, save_path):
    if not os.path.exists(save_path):
        response = requests.get(url)
        with open(save_path, "wb") as f:
            f.write(response.content)
    return save_path

# Download and load the model
model_path = download_model(model_url, model_path)
try:
    model = tf.keras.models.load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# File uploader for the image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((32, 32))  # CIFAR-10 images are 32x32 pixels
    image_array = np.array(image) / 255.0  # Normalize the image
    return np.expand_dims(image_array, axis=0)

# Function to plot the prediction result
def plot_prediction(image, prediction, class_names):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title(f"Predicted: {class_names[np.argmax(prediction)]} ({100 * np.max(prediction):.2f}%)")
    plt.axis('off')
    st.pyplot(plt)

# Process and classify the uploaded image
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    plot_prediction(image, prediction, class_names)
    st.write(f"Predicted label: {class_names[np.argmax(prediction)]} with {100 * np.max(prediction):.2f}% confidence")
