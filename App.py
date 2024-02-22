import streamlit as st
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

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
    

# Load the saved model from Google Drive: https://drive.google.com/file/d/1fbIzH3C-UXcjBYT3dkPqxv60o0y35pnq/view?usp=sharing    
weights_file_id = '1fbIzH3C-UXcjBYT3dkPqxv60o0y35pnq'
weights_url = f'https://drive.google.com/uc?id={weights_file_id}'

# st.write("Downloading weights file...")
gdown.download(weights_url, 'superhero_classifier_model_3.h5', quiet=True)
try:
    model = tf.keras.models.load_model("superhero_classifier_model_3.h5")
except Exception as e:
    st.error(f"Error loading the model: {e}")


# Streamlit UI
st.title("Superhero Image Classifier")

# Superhero names and corresponding image file names
superheroes = {
    'Batman': 'Batman.jpg',
    'Superman': 'Superman.webp',
    'Wonder Woman': 'Wonder Woman.jpg',
    'The Flash': 'The Flash.webp',
    'Black Panther': 'Black Panther.png',
    'Spiderman': 'Spiderman.jpg',
    'Iron Man': 'Iron Man.webp',
    'Captain America': 'Captain America.jpg',
    'Hulk': 'Hulk.jpg',
    'Black Widow': 'Black Widow.jpg',
}

col1, col2, col3, col4, col5 = st.columns(5)
base_image_path = os.path.join(os.getcwd(), 'Notebooks', 'UI Images')

with col1:
    for i in range(5):
        name, image_filename = list(superheroes.items())[i]
        image_path = os.path.join(base_image_path, image_filename)
        image = Image.open(image_path)
        st.image(image, width=29)
with col2:
    for i in range(5):
        name, image_filename = list(superheroes.items())[i]
        st.markdown(f"<p style='font-size: 18px;'>{name}</p>", unsafe_allow_html=True)
        # st.write(f"{name}")
        
with col4:
    for i in range(5, 10):
        name, image_filename = list(superheroes.items())[i]
        image_path = os.path.join(base_image_path, image_filename)
        image = Image.open(image_path)
        st.image(image, width=29)       
with col5:
    for i in range(5, 10):
        name, image_filename = list(superheroes.items())[i]
        st.markdown(f"<p style='font-size: 18px;'>{name}</p>", unsafe_allow_html=True)
        # st.write(f"{name}")

st.write()

allowed_extensions = ["jpg", "jpeg", "png", "webp"]
uploaded_file = st.file_uploader("Choose an Image...", type=allowed_extensions)

if uploaded_file is not None:
    # Preprocess to convert to 224*224
    image = Image.open(uploaded_file)
    image_array = preprocess_image(image, (224, 224))

    # Prediction
    predictions = model.predict(np.expand_dims(image_array, axis=0))
    predicted_class_value = np.argmax(predictions)
    predicted_class_name = name_label_mappings.get(predicted_class_value, "Unknown")

    # Show results
    st.write("Predicted Superhero:", predicted_class_name)
    st.write("Probability:", np.max(predictions))
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)