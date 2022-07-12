import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image, ImageOps
import os
import streamlit as st
from filesplit.merge import Merge
import pillow_heif

# Settings
st.set_page_config(page_title="Herbarium Classification", page_icon=None, 
                  layout="wide", 
                  initial_sidebar_state="auto", 
                  menu_items=None)


# Helpers ---
def prep_input_image(image, resize_factor=(380, 380)):
    image = image.astype('float32')
    # Normalize to ImageNet (the base model of the ResNet) 
    #image -= tf.constant([0.485, 0.456, 0.406], shape=[1, 1, 3], dtype=image.dtype)
    #image /= tf.constant([0.229, 0.224, 0.225], shape=[1, 1, 3], dtype=image.dtype)
    # Crop towards center of image (crop out borders)
    #image = tf.image.central_crop(image, central_fraction = 0.9)
    # Resize
    image = np.array(image)
    image = cv2.resize(image, resize_factor, interpolation=cv2.INTER_LINEAR)
    # Rescale pixels
    image *= 1.0/255

    # Reshape dimensions
    image = image.reshape(1, image.shape[0], image.shape[1], 3)

    return image

def top_5_predictions(pred):
    # Create dictionary maps to correct for differences in meta_data and model trained data
    map_name_to_cat_id = dict(zip(meta_data.scientific_name, meta_data.category))
    map_label_to_name = dict(zip(range(15501), sorted(set(meta_data.scientific_name))))
    # Top 5 class prediction
    num_class = 15501
    pred = pred.reshape(num_class)
    pred_idx = np.argpartition(pred, -5)[-5: ]

    # Map to get true category labels
    pred_class = np.array([map_name_to_cat_id[map_label_to_name[p]] for p in pred_idx])
    # Probabilities for top 5 preds
    pred_prob = pred.reshape(num_class)[pred_idx]

    image_guess = pd.DataFrame({
    'class' : pred_class,
    'prob' : pred_prob
    }).sort_values(by = 'prob', ascending=False)
    sorted_classes = [c for c in image_guess['class']]

    return image_guess, sorted_classes

def query_plant_info(categories, X=None,y=None):
    
    # Locate row
    rows = meta_data.loc[meta_data.category.isin(categories)]
    # Extract info
    category = [c for c in rows['category']]
    scientific_name = [sc for sc in rows['scientific_name']]
    family = [f for f in rows['family']]
    genus = [g for g in rows['genus']]
    species = [s for s in rows['species']]
    images = []
    # plot example image
    if X is not None:
        for categ in [c for c in rows['category']]:
            img = X[np.where(y==categ)[0][0]]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            
            #plt.imshow(img)
            #plt.axis('off')
            #plt.show()
    
    return category, scientific_name, family, genus, species#, images
#query_plant_info(pred_classes)


def upload_predict(image, model):
        size = (380,380)    
        image = np.asarray(image)
        if len(image.shape) > 2 and image.shape[2] == 4:
            #convert the image from RGBA2RGB (for example, if input is PNG)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image = prep_input_image(image)
        prediction = model.predict(image)
        top5, pred_classes = top_5_predictions(prediction)
        pred_prob = [p for p in top5['prob']]
        category, scientific_name, family, genus, species = query_plant_info(pred_classes)
        out = pd.DataFrame({
            'category_id' : category,
            'confidence' : pred_prob,
            'scientific name' : scientific_name,
            'family' : family,
            'genus' : genus,
            'species' : species
        })

        return out

@st.cache(allow_output_mutation=True)
def load_model():
    # Load model structure
    ## Load in JSON model config
    json_file = open('./data/herb-model.json', 'r')
    loaded_json_mod = json_file.read()
    json_file.close()
    ## Feed in JSON config to keras model
    model = tf.keras.models.model_from_json(loaded_json_mod)

    # Load model weights
    ## Concatenate split weights into one file (creates full_weights.h5)
    merge = Merge('./data/', './data/', 'full_weights.h5')
    merge.merge(cleanup=False)   ##keep split filesin data/ dir with False
    ## Load saved weights into model
    model.load_weights('./data/full_weights.h5')

    return model

#-----#
# APP #--------------------------------------------------------
#-----#

with st.spinner('Loading trained model...'):
    model = load_model()
    # Load metadata
    meta_data = pd.read_csv('./data/herb22meta_data.csv')
    # Load Image Arrays and Corresponding Labels for Examples
    #pp = "C:/Users/hanse/Documents/herbarium22/data/"
    #with open(pp+'image-arrays.npy', 'rb') as f:
    #    X = np.load(f)
    #with open(pp+'labels.npy', 'rb') as f:
    #    y = np.load(f)

# INTRO
st.write("""
         # North American Vascular Plant Classification
         ### The Herbarium 2022: Flora of North America
         *DISCLAIMER:* This app is built on an image classification model which was trained on the 
         [Herbarium 2022](https://www.kaggle.com/competitions/herbarium-2022-fgvc9) dataset.  
         **The model is trained to predict the species of any North American vascular land plant**
         (which includes lycophytes, ferns, gymnosperms, and flowering plants). It cannot possibly identify any other species.  
         That said, the *15,501 [vascular plants](https://en.wikipedia.org/wiki/Vascular_plant)* which were included in the training data 
         constitute more than 90\% of the taxa documented in North America.
         The plant images used to train this model (examples shown below) come from [herbariums](https://en.wikipedia.org/wiki/Herbarium) 
         from around the world, and they generally position the plant in front of a white background, with decent lighting, and minimal clutter.
         Therefore, plant images taken in the wild are likely to be harder to classify correctly.  
         (For accurate classifications, check out apps like PictureThis.)  

         The deep learning model used for this project is a ResNet50 model pre-trained on ImageNet and then trained for 10
         epochs on the Herbarium 2022 dataset. It achieved an F1-score of 0.73 on the Herbarium test data available on Kaggle.  
         """
         )


# FILE UPLOADER
file = st.file_uploader("Upload an image of a North American vascular land plant.", 
                        help="Supported filetypes: jpg/jpeg, png, heic (iPhone).") #type=["jpg", "png", "heic, "])
st.set_option('deprecation.showfileUploaderEncoding', False)


# EXAMPLE IMAGES
st.write("""
        ### Example Images  
        [Source.](https://www.kaggle.com/competitions/herbarium-2022-fgvc9) 
        """)
ip = "./example_images/"
paths = [ip+'ex_img_200.jpg', ip+'ex_img_499.jpg', ip+'ex_img_640.jpg']
ex_imgs = [Image.open(paths[0]), Image.open(paths[1]), Image.open(paths[2])]
st.image(ex_imgs, caption=['Aeschynomene viscidula Michx.',
                            'Alternanthera philoxeroides (Mart.) Griseb.',
                            'Amsonia peeblesii Woodson'],
                            use_column_width=False, width=400)

# PROCESS IMAGE AND PREDICT
if file is None:
    st.text("Upload image for prediction.")
else:
    bytes_data = file.read()
    filename = file.name
    # If file is in HEIC format (ie, if uploaded from iphone)
    if filename.split('.')[-1] in ['heic', 'HEIC', 'heif', 'HEIF']:
        heic_file = pillow_heif.read_heif(file)
        img = Image.frombytes(
            heic_file.mode,
            heic_file.size,
            heic_file.data
        )
    else:
        img = Image.open(file)
    st.write("### Your Image:")
    st.write("filename:", filename)
    st.image(img, width=400, use_column_width=False)
    pred_classes = upload_predict(img, model)
    st.write("# Prediction")
    st.write(pred_classes)
    st.write("""
            The confidence scores are the model's predicted probabilities for each class.
            Shown above are the top model's 5 guesses for your image, sorted by highest condifence.  
            The scores are generally fairly low, but this is partially due to the large number of possible classes.  
            """)


st.write("---")
st.write("By [Hans Elliott](https://hans-elliott99.github.io/)")

