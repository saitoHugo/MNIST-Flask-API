"""
Script de API que tem a rota de predição. Recebe uma image e retorna a classificação
"""


import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, Response, jsonify
import json

app = Flask(__name__)


@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        
        ###############################################
        ################# PREPROCESSING ###############
        ###############################################
        
        #Collection the data from requests
        data = request.data
        #print("Data collected from request: ")
        #print(data)
        
        #Converting into np from string
        img_np = np.fromstring(data, np.uint8)
        #print("Data converted to from string to 'np.uint8': ")
        #print(img_np)
        
        #Decoding the image
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        #print("Image Decoded: ")
        #print(img)
        #print("Shape of Decoded Image: ", img.shape)
        
        #Convert to Gray Scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #print("Gray Scaled Image: ")
        #print(img)
        #print("Shape of Gray Scaled Image: ", img.shape)
        #print("Type of Gray Scale Image: ", type(img))
        
        #Resize and New Axis
        img = cv2.resize(img, (28, 28))
        #Adding new axis (input format)
        img = img[np.newaxis, ...]
        #print('Image Resized: ', img)
        #print('Image Resized Shape: ',img.shape)
        #print('Type of Resized Image: ', type(img))
        
        #Normalization
        img = img/255.0
        #print('Normalized Image Valus: ', img)
        #print('Normalized Image Shape: ', img.shape)
        #print ('Type of Normalized Image file: ', type(img))
        
        
        
        ###############################################
        ################# PREDICTION ##################
        ###############################################        
        
    
        #Predicting the class
        #prediction = model.predict_classes(img)
        prediction = model.predict_classes(img)
        #print ('Prediction: ', prediction)
        
        #Cleaning the result and convert to str
        prediction = str(prediction[0])
        print("String of Prediction: ", prediction)
        #Convert to json (dict) format
        response_dict = {'message': prediction}
        js_dump = json.dumps(response_dict)
        
        #Formatting the response message
        response = Response(js_dump,status=200, mimetype='application/json')
        
        return response
        



if __name__ == '__main__':
    
    print("* Loading Keras model and Flask starting API...")
    
    #Loading the model saved befone (.h5)
    global model
    model = tf.keras.models.load_model('model/MNIST.h5')
    print(model.summary())
    print("Model loaded!")
    
    #Run the Web App
    app.run(debug = True, host='0.0.0.0', port=5050)







####
#####load the model
####model = tf.keras.models.load_model('MNIST.h5')
####model.summary()
####
####
#####load the test image
####img = Image.open('test_img.jpg')
####img = np.asarray(img.getdata())
####print('Image: ', img.resize(1, 28, 28))
####print('Image Shape: ',img.shape )
####print("Image Values: ", img)
####img = img/255.0
####print('Normalized Image Valus: ', img)
####print('Normalized Image Shape: ', img.shape)
####print ('Type of Image file: ', type(img))
####
####
#####Prediction
#####prediction = model.predict_classes(img)
####prediction = model.predict(img)
####print ('Prediction: ', prediction)
####