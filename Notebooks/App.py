import streamlit as st
import requests
import numpy as np
from PIL import Image
import tensorflow as tf

def preprocess_image(img, img_size):
    img_resize = img.resize(img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img_resize)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    return img_array

# Label mappings from Model Training.ipynb
label_mappings = {
                   'Hulk': 0
                 , 'Batman': 1
                 , 'Iron Man': 2
                 , 'Spiderman': 3
                 , 'Superman': 4
                 , 'The Flash': 5
                 , 'Wonder Woman': 6
                 , 'Captain America': 7
                 , 'Black Panther': 8
                 , 'Black Widow': 9
                 }

name_label_mappings = {value: name for name, value in label_mappings.items()}
    
# Load the saved model
model = tf.keras.models.load_model("superhero_classifier_model_3.h5")

# Streamlit UI
st.title("Superhero Image Classifier")

uploaded_file = st.file_uploader("Choose an Image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)

    # Preprocess to convert to 224*224
    image_array = preprocess_image(image, (224, 224))

    # Prediction
    predictions = model.predict(np.expand_dims(image_array, axis=0))
    predicted_class_value = np.argmax(predictions)
    predicted_class_name = name_label_mappings.get(predicted_class_value, "Unknown")

    # Show results
    st.write("Predicted Superhero:", predicted_class_name)
    st.write("Probability:", np.max(predictions))
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)