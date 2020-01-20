
#""""
#Script para test da API de predição com upload de uma imagem local e predição com resposta
#""""

from __future__ import print_function
import requests
import json
import cv2



addr = 'http://localhost:5050'
test_url = addr + '/predict'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('images/test93.jpg')
cv2.imshow('image', img)
# encode image as jpeg
#print (img)
_, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
print("Response received: ")
print(response)

#Acessing the predicted class
print("Predicted class: ")
predicted_class = json.loads(response.text)['message']
print(predicted_class)
# decode response
#print(json.loads(response.text))

# expected output: {u'message': u'image received. size=124x124'}