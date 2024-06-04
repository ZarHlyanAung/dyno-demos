import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import zipfile
import io
import os

# Streamlit app title and description
st.title("Image Classification App")
st.write("Upload a ZIP file containing the model and an image to classify.")

# File uploader for the zip file containing the model
uploaded_zip = st.file_uploader("Choose a ZIP file with the model...", type="zip")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Temporary directory to extract the model
temp_model_dir = "/tmp/model_dir"

def extract_model(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_model_dir)

def preprocess_image(image):
    image = image.resize((32, 32))  # CIFAR-10 images are 32x32 pixels
    image_array = np.array(image) / 255.0  # Normalize the image
    return np.expand_dims(image_array, axis=0)

def plot_prediction(image, prediction, class_names):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title(f"Predicted: {class_names[np.argmax(prediction)]} ({100 * np.max(prediction):.2f}%)")
    plt.axis('off')
    st.pyplot(plt)

if uploaded_zip and uploaded_image:
    extract_model(uploaded_zip)
    
    # Load the model
    model = tf.keras.models.load_model(temp_model_dir)
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Process the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    plot_prediction(image, prediction, class_names)
    st.write(f"Predicted label: {class_names[np.argmax(prediction)]} with {100 * np.max(prediction):.2f}% confidence")
