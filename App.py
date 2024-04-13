import streamlit as st
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os
import gc

def preprocess_image(img, img_size):
    img_resize = img.resize(img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img_resize)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    return img_array


## Load the Model from Google Drive
save_path = 'superhero_classifier_model_3.h5'

@st.cache_resource(ttl = 60*60 *24 *7, show_spinner="Fetching model from cloud...")
def load_model_from_google_drive(fileid, save_path):
    # Download and Save Model
    weights_url = f'https://drive.google.com/uc?id={fileid}'
    gdown.download(weights_url, save_path, quiet=True)
    # Load and Return Model
    try:
        model_gd = tf.keras.models.load_model(save_path)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
    return model_gd

google_drive_file_id = st.secrets["GOOGLE_DRIVE_FILE_ID"]
model_gd = load_model_from_google_drive(google_drive_file_id, save_path)


## Load the saved model from Dropbox
# def load_model_from_dropbox(dropbox_link, save_path):
#     # Download and Save Model
#     response = requests.get(dropbox_link)
#     with open(save_path, 'wb') as f:
#         f.write(response.content)
#     # Load and Return Model        
#     try:
#         model_dbx = tf.keras.models.load_model(save_path)
#     except Exception as e:
#         st.error(f"Error loading the model: {e}")
#     return model_dbx

# dropbox_link = st.secrets["DROPBOX_LINK"]
# model_dbx = load_model_from_dropbox(dropbox_link, save_path)


## Load the saved model from local device
# model_lcl = tf.keras.models.load_model("Notebooks/superhero_classifier_model_3.h5")

## Selecting Cloud to load model from
model = model_gd


# Garbage Collector to manage memory in Streamlit    
gc.enable()



# Streamlit UI
st.title("Superhero Image Classifier")

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


# Garbage Collection
gc.collect()