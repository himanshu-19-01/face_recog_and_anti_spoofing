import random 
import cv2
import imutils
import f_liveness_detection
import questions
import os
from imutils.video import VideoStream
import face_recognition
import tensorflow as tf
import numpy as np 
import argparse
import imutils
import pickle
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from face_anti_spoofing import show
import base64
from welcome_screen_with_desired_output import face_recog

import json
import requests
import base64


# from new_api import my_api



def run(model_path, le_path, detector_folder, confidence=0.5):
    r=0
    args = {'model':model_path, 'le':le_path, 'detector':detector_folder, 'confidence':confidence}

    print('[INFO] loading face detector...')
    proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
    model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
    detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    
    # load the liveness detector model and label encoder from disk
    liveness_model = tf.keras.models.load_model(args['model'])
    le = pickle.loads(open(args['le'], 'rb').read())
    
    # initialize the video stream and allow camera to warmup
    print('[INFO] starting video stream...')
    vs = cv2.VideoCapture(0)
    
    sequence_count = 0 
    
    # iterate over the frames from the video stream
    while True:
        
        _,frame = vs.read()
        frame = cv2.flip(frame, 1)
        # frame = imutils.resize(frame,height=768,width=)
        
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        detector_net.setInput(blob)
        detections = detector_net.forward()
        
        # iterate over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e. probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
            
            # filter out weak detections
            if confidence > args['confidence']:
                # compute the (x,y) coordinates of the bounding box
                # for the face and extract the face ROI
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')
                
                # expand the bounding box a bit
                # (from experiment, the model works better this way)
                # and ensure that the bounding box does not fall outside of the frame
                startX = max(0, startX-20)
                startY = max(0, startY-20)
                endX = min(w, endX+20)
                endY = min(h, endY+20)
                
                # extract the face ROI and then preprocess it
                # in the same manner as our training data
                face = frame[startY:endY, startX:endX] # for liveness detection
                # expand the bounding box so that the model can recog easier
                face_to_recog = face # for recognition
                # some error occur here if my face is out of frame and comeback in the frame
                try:
                    face = cv2.resize(face, (32,32)) # our liveness model expect 32x32 input
                except:
                    break
            
                # face recognition
                rgb = cv2.cvtColor(face_to_recog, cv2.COLOR_BGR2RGB)
                #rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb)
                # initialize the default name if it doesn't found a face for detected faces
                            
                face = face.astype('float') / 255.0 
                face = tf.keras.preprocessing.image.img_to_array(face)
            
                face = np.expand_dims(face, axis=0)
            
                preds = liveness_model.predict(face)[0]
                j = np.argmax(preds)
                label_name = le.classes_[j] # get label of predicted class
                
                # draw the label and bounding box on the frame
                label = f'{label_name}: {preds[j]:.4f}'
                if label_name == 'fake':
                    sequence_count = 0
                else:
                    sequence_count += 1
                print(f'[INFO] {label_name}, seq: {sequence_count}')
                
                if label_name == 'fake':
                    print("fake")
                    detect = "fake"
                    cv2.putText(frame, "Don't try to Spoof !", (startX, endY + 25), 
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
                    # exit()
                
                # cv2.putText(frame, name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,130,255),2 )
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 4)
                if label_name == 'real':
                    detect = "real"
                    print("real")
                    # retval, buffer = cv2.imencode('.jpg', frame)
                    # jpg_as_text = base64.b64encode(buffer)
                    cv2.imwrite("./faces/real.jpg",frame)
                    
                # print(jpg_as_text)
            # my_api(my_string1,my_string2)
        # show the output fame and wait for a key press
        cv2.imshow('Video', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or sequence_count==30:
            continue
        r = r+1
        if r==30:
            #welcome screen function
            face_recog(vs)
            break
        
    # cleanup
    # vs.stop()
    cv2.destroyAllWindows()
    

    time.sleep(2)

    return label_name

   

if __name__ == '__main__':
    label_name = run('liveness.model', 'label_encoder.pickle', 
                                            'face_detector', confidence=0.5)
    



