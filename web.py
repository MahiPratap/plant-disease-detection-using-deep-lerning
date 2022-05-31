# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 18:06:38 2021

@author: MAHI
"""
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model_inception_segmented.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds == 0:
        preds= "Apple___Apple_scab" # if index 0 
    elif preds == 1:
        preds= 'Apple___Black_rot' # # if index 1
    elif preds == 2:
        preds='Apple___Cedar_apple_rust' # if index 2  
    elif preds ==3:
        preds="Apple___healthy" # if index 3
    elif preds ==4:
        preds="Blueberry___healthy"
    elif preds ==5:
        preds="Cherry_(including_sour)___Powdery_mildew"
    elif preds ==6:
        preds="Cherry_(including_sour)___healthy"
    elif preds ==7:
        preds= "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"
    elif preds ==8:
        preds= "Corn_(maize)___Common_rust_"
    elif preds ==9:
        preds= "Corn_(maize)___Northern_Leaf_Blight"
    elif preds ==10:
        preds= "Corn_(maize)___healthy"
    elif preds ==11:
        preds="Grape___Black_rot"
    elif preds ==12:
        preds= "Grape___Esca_(Black_Measles)"
    elif preds ==13:
        preds= "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"
    elif preds ==14:
        preds="Grape___healthy"
    elif preds ==15:
        preds= "Orange___Haunglongbing_(Citrus_greening)"
    elif preds==16:
        preds= "Peach___Bacterial_spot"
    elif preds ==17:
        preds="Peach___healthy"
    elif preds ==18:
        preds="Pepper,_bell___Bacterial_spot"
    elif preds ==19:
        preds= "Pepper,_bell___healthy"
    elif preds ==20:
        preds= "Potato___Early_blight"
    elif preds ==21:
        preds= "Potato___Late_blight"
    elif preds ==22:
        preds="Potato___healthy"
    elif preds ==23:
        preds="Raspberry___healthy"
    elif preds ==24:
        preds="Soybean___healthy"
    elif preds ==25:
        preds= "Squash___Powdery_mildew"
    elif preds ==26:
        preds= "Strawberry___Leaf_scorch"
    elif preds ==27:
        preds= "Strawberry___healthy"
    elif preds ==28:
        preds= "Tomato___Bacterial_spot"
    elif preds ==29:
        preds= "Tomato___Early_blight"
    elif preds ==30:
        preds= "Tomato___Late_blight"
    elif preds ==31:
        preds= "Tomato___Leaf_Mold"
    elif preds ==32:
        preds= "Tomato___Septoria_leaf_spot"
    elif preds ==33:
        preds= "Tomato___Spider_mites Two-spotted_spider_mite"
    elif preds ==34:
        preds= "Tomato___Target_Spot"
    elif preds ==35:
        preds= "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
    elif preds ==36:
        preds= "Tomato___Tomato_mosaic_virus"
    else:
        preds= "Tomato___healthy"
        
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)