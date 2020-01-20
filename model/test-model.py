"""
Script de test do modelo que foi treinado, apenas carrega o modelo e tenta fazer uma predição com uma imagem local

"""

from PIL import Image
import numpy as np
import tensorflow as tf


#load the model
model = tf.keras.models.load_model('MNIST.h5')
model.summary()


#load the test image
img = Image.open('test_img.jpg')
img = np.asarray(img.getdata())
print('Image: ', img.resize(1, 28, 28))
print('Image Shape: ',img.shape )
print("Image Values: ", img)
img = img/255.0
print('Normalized Image Valus: ', img)
print('Normalized Image Shape: ', img.shape)
print ('Type of Image file: ', type(img))


#Prediction
#prediction = model.predict_classes(img)
prediction = model.predict(img)
print ('Prediction: ', prediction)