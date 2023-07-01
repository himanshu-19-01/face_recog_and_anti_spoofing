#libraries
from flask import Flask,jsonify
import io, base64
from PIL import Image

from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import requests
import io, base64
from PIL import Image
from keras.models import model_from_json
import json
from flask import Flask, request, jsonify
import cv2

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

model.load_weights('vgg_face_weights.h5')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    print(img)
    return img

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def stringToRGB(base64_string):
    decoded_data=base64.b64decode((base64_string))
    img_file = open('./faces/image.jpeg', 'wb')
    img_file.write(decoded_data)
    img_file.close()

def stringToRGBB(base64_string):
    decoded_data=base64.b64decode((base64_string))
    img_file = open('./faces/image1.jpeg', 'wb')
    img_file.write(decoded_data)
    img_file.close()
    


vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

epsilon = 0.46

app=Flask(__name__)

@app.route('/verifyFace',methods=['GET','POST'])

def verifyFace():
    data_dict = json.loads(request.data.decode('utf-8'))
    aadharimg = data_dict['a']
    stringToRGB(aadharimg)
    img1 = "image.jpeg"
    modelimg = data_dict['b']
    stringToRGBB(modelimg)
    img2 = "real.jpg"
    img1_representation = vgg_face_descriptor.predict(preprocess_image('static/images/faces/%s' % (img1)))[0,:]
    img2_representation = vgg_face_descriptor.predict(preprocess_image('static/images/faces/%s' % (img2)))[0,:]
    
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    
    print("Cosine similarity: ",cosine_similarity)
    cosine_percentage = (1 - cosine_similarity)*100
    print("Percentage: ",cosine_percentage)

    if(cosine_similarity < epsilon):#static    static\images\faces\image.jpeg
        result = "success"
        print("They are same person")
        output = {'Result':result,'percentage':cosine_percentage}
        return jsonify(output)
        
    else:
        print("They are not same person!")
        result = "failure"
        output = {'Result':result,'percentage':cosine_percentage}
        return jsonify(output)

if __name__ == '__main__':
    app.run(port=9999)