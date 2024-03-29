# HeroClassifier

HeroClassifier is a deep learning project designed for classifying images of 10 superheroes. The repository encompasses various components such as data scraping, data preprocessing, model development utilizing transfer learning (VGG16), training and evaluation scripts, and model deployment using Streamlit. The model demonstrates an impressive 88.67% accuracy on unseen data. The application is hosted [here](https://heroclassifier.streamlit.app/)

## Project Structure

- **Images Folder:**
  Contains images of all 10 superheroes organized into different folders, with each superhero having 154-155 images.

- **Notebooks Folder:**
  This folder houses the core code for the project:

  - `Data Scraping and Preprocessing.ipynb`: Web scraping of images and preprocessing the scraped images to align with the base model VGG16.

  - `Model Training.ipynb`: Notebook contains code for CNN training and saving the model weights.

  - `App.py`: Streamlit code for a user-friendly UI to upload images of any superhero from the 10 classes and make predictions. The weights are downloaded from the Dropbox


## Superheroes Classes

The model is capable of classifying images belonging to the following superheroes:

1. Batman
2. Black Panther
3. Black Widow
4. Captain America
5. Hulk
6. Iron Man
7. Spiderman
8. Superman
9. The Flash
10. Wonder Woman


## Streamlit Usage

In case of encountering any issues while uploading images through the Streamlit UI, you can disable the XSRF protection using the following command:

```bash
streamlit run app.py --server.enableXsrfProtection false
```

