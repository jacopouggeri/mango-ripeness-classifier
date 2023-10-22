
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import os

from sklearn.utils import shuffle

import tensorflow as tf


# random seeds for reproducibility
tf.random.set_seed(123)

from tensorflow.keras.models import load_model

loaded_model = load_model('CNN_model.h5')

def hard_predict(array):
    probability = loaded_model.predict(array)
    
    # Convert soft probability to hard prediction
    hard_pred = int(probability >= 0.5)

    return hard_pred

def resize_image(image_array, target_size=(640, 640)):

    # Resizing the image
    resized_image = cv2.resize(image_array, target_size, interpolation=cv2.INTER_LINEAR)
    img_with_batch_dimension = np.expand_dims(resized_image, axis=0)
    return img_with_batch_dimension

def image_process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=resize_image(img)
    return img

def prediction_output(img):
    # Read the image
    img = image_process(img)
    # Predict the class
    prediction = hard_predict(img)
    if prediction == 1:
        return "ripe"
    else:
        return "unripe"