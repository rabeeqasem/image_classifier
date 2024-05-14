# Our suggestion is to create a module just for utility functions like preprocessing images. Make sure to include all files necessary to run the predict.py file in your submission.

# The predict.py module should predict the top flower names from an image along with their corresponding probabilities.

# Basic usage: python $ python predict.py /path/to/image saved_model

# Options:

# --top_k : Return the top 
# 𝐾
# K most likely classes:
# $ python predict.py /path/to/image saved_model --top_k $$K$$
# --category_names : Path to a JSON file mapping labels to flower names:
# $ python predict.py /path/to/image saved_model --category_names map.json


# TODO: Make all necessary imports.
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
tfds.disable_progress_bar()
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import argparse
from PIL import Image
import json
import tqdm
import os


def process_image(image):

    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image.numpy()

def predict(image_path, model,class_names, top_k=5):

    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    processed_test_image = np.expand_dims(processed_test_image, axis=0)
    ps = model.predict(processed_test_image)
    top_k_values, top_k_indices = tf.nn.top_k(ps, k=top_k)
    top_k_values = top_k_values.numpy().squeeze()
    top_k_indices = top_k_indices.numpy().squeeze()
    return top_k_values, top_k_indices

def visualize(test_image,top_k_values, top_k_indices, class_names):
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(test_image)
    ax1.axis('off')
    ax1.set_title(class_names[str(1)])
    ax2.barh(np.arange(5), top_k_values)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(5))
    ax2.set_yticklabels([class_names[str(index+1)] for index in top_k_indices], size='small')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict flower name from image')
    parser.add_argument('-i', '--image_folder', type=str, help='Path to image file')
    parser.add_argument('-m', '--model', type=str, help='Path to model file')
    parser.add_argument('-k', '--top_k', type=int, help='Return the top K most likely classes')
    parser.add_argument('-c', '--category_names', type=str, help='Path to a JSON file mapping labels to flower names')
    parser.add_argument('-v', '--visualize', type=bool,default=True, help='Visualize the image with the top k classes probabilities')


    args = parser.parse_args()
    image_path = args.image_folder
    model_path = args.model
    top_k = args.top_k
    category_names = args.category_names

    print('Image Path:', image_path)



    model=tf.keras.models.load_model(model_path)
    

    with open(category_names, 'r') as f:
        class_names = json.load(f)

    

    print('Predicting image:', image_path)
    top_k_values, top_k_indices = predict(image_path, model, class_names, top_k=5)
    print('Probabilities:', top_k_values)
    print('Classes:', top_k_indices)
    print('Class Names:', [class_names[str(index+1)] for index in top_k_indices])

    if args.visualize:
        im = Image.open(image_path)
        test_image = np.asarray(im)
        visualize(test_image,top_k_values, top_k_indices, class_names)
    else:
        print('if you want to visualize the image with the top k classes probabilities, set the visualize flag to True')


