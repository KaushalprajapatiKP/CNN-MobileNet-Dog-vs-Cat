import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import random
import os
from PIL import Image
import tf_keras as keras
import tensorflow_hub as hub
from tensorflow.keras.models import load_model


import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model

def load_my_model():
    model = load_model("cat_vs_dog_mobilenet_model.keras", custom_objects={'KerasLayer': hub.KerasLayer})
    return model

model = load_my_model()
# Function to preprocess image using OpenCV
def preprocess_image(image_path):
    input_image = cv2.imread(image_path)
    input_image_resize = cv2.resize(input_image, (224, 224))
    input_image_scaled = input_image_resize / 255.0  # Normalize to [0, 1]
    image_reshaped = np.reshape(input_image_scaled, [1, 224, 224, 3])
    return input_image, image_reshaped

# Prediction function
def predict_image(image_array):
    input_prediction = model.predict(image_array)
    input_pred_label = np.argmax(input_prediction)
    return input_pred_label

# Streamlit App UI
st.title("Cat vs Dog Classifier")

# User input: upload or random selection
option = st.radio("Choose an option:", ("Upload an image", "Select a random image from test set"))

if option == "Upload an image":
    uploaded_file = st.file_uploader("Upload a dog or cat image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Save uploaded image temporarily
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.read())
        
        # Preprocess the image
        input_image, image_array = preprocess_image("temp_image.jpg")
        
        # Display the uploaded image
        st.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
        
        # Predict
        prediction = predict_image(image_array)
        if prediction == 0:
            st.write("Prediction: **The image represents a Cat**")
        else:
            st.write("Prediction: **The image represents a Dog**")

elif option == "Select a random image from test set":
    test_dir = "E:\Deep  learning\CNN-MobileNet-Dog-vs-Cat\\test_set"  # Replace with your test set directory
    all_images = []
    for label in ["cat", "dog"]:
        label_dir = os.path.join(test_dir, label)
        all_images.extend([os.path.join(label_dir, img) for img in os.listdir(label_dir)])
    
    random_image_path = random.choice(all_images)
    
    # Preprocess the random image
    input_image, image_array = preprocess_image(random_image_path)
    
    # Display the random image
    st.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), caption="Randomly Selected Image", use_column_width=True)
    
    # Predict
    prediction = predict_image(image_array)
    if prediction == 0:
        st.write("Prediction: **The image represents a Cat**")
    else:
        st.write("Prediction: **The image represents a Dog**")
