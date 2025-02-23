import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Loading the saved best model
@st.cache(allow_output_mutation=True)
def load_trained_model(best_model):
    model = tf.keras.models.load_model(best_model)
    return model

st.title("Fish Species Classification")

st.write("""
Upload an image of a fish and let the model predict which species it belongs to!
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Converting the file to an opencv image.
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Resizing the image to the same size as the model input
    img_height, img_width = 224, 224
    image = image.resize((img_width, img_height))
    
    # Preprocess: Converting to numpy array & scaling
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # shape (1, 224, 224, 3)

    # Loading model
    model_path = 'best_model.h5'  # Ensuring the path is correct
    model = load_trained_model(model_path)

    # Predicting
    predictions = model.predict(image_array)
    st.write("Predictions:", predictions)

    # Retrieving the index of highest probability
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Getting class names 
    class_names = ['animalfish','animalfishbass','blackseasprat','giltheadbream','hoursemackerel','redmullet','redseabream','seabass','shrimp',
                   'stripedredmullet','trout']  
    predicted_class = class_names[predicted_class_index]
    
    confidence_score = np.max(predictions)*100

    st.success(f"**Predicted Fish Category:** {predicted_class}")
    st.info(f"**Confidence Score:** {confidence_score:.2f}%")
