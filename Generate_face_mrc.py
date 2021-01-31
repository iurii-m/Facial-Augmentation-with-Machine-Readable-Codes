# -*- coding: utf-8 -*-
"""
Console application to create MRC (QR Code) from the input image or webcam capture.

@author: iurii
"""


import numpy as np
import argparse
import os
import cv2

from base64_to_array import array_2_base64
from base64_to_array import base64_2_array

from mtcnn import MTCNN

from alignment_frontal import _align_faces
from alignment_frontal import _read_image_2_cvMat

import glob
import os

import time

from vae_encode_face import encode_faces
from vae_encode_face import decode_faces

import qrcode
from PIL import Image

def detect_frontal_faces(image, detector, face_cascade, face_scale, face_size, name_id, datapath='data/aligned_test_face_src_images'):
    #detecting and preprocessing faces on the frame
    faces_to_encode = []
    #
    result, number_of_detected, main_idx = _align_faces(image, detector , face_scale, face_size)
    index=0
    #checking all the detected faces if they are frontal
    for res in result:
        # Convert into grayscale
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces_rects = face_cascade.detectMultiScale(gray, 1.1, 3)

        #Append if frontal
           
        if len(faces_rects)>0:
            faces_to_encode.append(res)
            cv2.imwrite(datapath+'/'+name_id+'_'+str(index)+'.png',res)
            index=index+1

    return (faces_to_encode)

def encode_faces_mrcs(encoded_faces, name_id, datapath='data/generated_QR_Codes'):
    
    index=0
     #save qr codes with timestamps names
    for encoded_face in encoded_faces:
        img = qrcode.make(array_2_base64(encoded_face))
        
        #save image
        img.save(datapath+'/'+name_id+'_'+str(index)+'.png')
        index=index+1
        

    return


if __name__ == '__main__':
    print('Launch application to encode face to MRC')
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--file', 
                        help='Image file from the folder. Works only when is used. Ex:1.jpg.')
    parser.add_argument('-d', '--if_detect_face', 
                        help='Binary value.  Define if face preprocessing is required', 
                        default=True)
    parser.add_argument("-w", "--webcam", 
                        help="Load h5 or tf model trained weights. By default is 0",
                        default=0)
    

    
    args = parser.parse_args()
    
    
    #define mtcnn face detector
    detector = MTCNN()
    face_scale= 2.0
    face_size=(128,128)

    # Load the cascade face detector
    face_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface_improved.xml')
    

    if args.file:
                
        #read image
        image = _read_image_2_cvMat(args.file)
        
        #encode input file to MRC    
        faces_2_encode = []
        
        #detect faces or take resized input image
        if args.if_detect_face:
            faces_2_encode = detect_frontal_faces(image, detector,face_cascade, face_scale, face_size, args.file[:-4], datapath='data/aligned_test_face_src_images')
        else:
            faces_2_encode.append(cv2.resize(image, face_size))
            cv2.imwrite('data/aligned_test_face_src_images'+'/'+args.file[:-4]+'.png',cv2.resize(image, face_size))
            
        
        #encode faces with vae
        faces_encoded = encode_faces(faces_2_encode, args)
        #encode mrcs
        encode_faces_mrcs(faces_encoded,args.file[:-4])
        print (len(faces_encoded), 'faces are encoded succesfully. Going ahead.')
        
        
        #regenerate faces with vae
        faces_decoded = decode_faces(faces_encoded, args)
        index=0
        #checking all the detected faces if they are frontal
        for face_decoded in faces_decoded:
            cv2.imwrite('data/regenerated_test_face_src_images'+'/'+args.file[:-4]+'_'+str(index)+'.png',(face_decoded* 255.0).astype(int))
            index=index+1
        print (len(faces_encoded), 'faces are regenerated succesfully. Going ahead.')
        
        print('File ', str(args.file), 'is processed succesfully. Quit the application.' )
    
    else:
        #load webcam to capture face image
        print('Load webcam to capture face image')
        
        #press 'esc' or 'q' to quit application
        print('Press ESC or Q to quit application')
           
        #Try toad video stream from default webcam
        #try:
        camera = cv2.VideoCapture(args.webcam)
        ret, frame = camera.read()
        
        print('Webcam launched. Look at the camera and Press C key.')
        
        #analyzing frames
        while ret:
            ret, frame = camera.read()
            
            #timestamp
            ts = str(time.time())
            
            #try fast detect face with cascade detectors
            if_face_detected=False
            
            #detect face with lbp cascade on the mtcnn main detected face
            # Convert into grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            if(len(faces)>0):
                if_face_detected=True
            
            #draw faces on the frame
            frame_show = frame.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(frame_show, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.imshow('webcam',frame_show)
            
            #launching encoding if pressed c key
            pressed_key = cv2.waitKey(2)
            if pressed_key == ord('c') :
                if if_face_detected :
                        

                    #detect frontal faces
                    faces_2_encode = detect_frontal_faces(frame, detector, face_cascade , face_scale, face_size, ts, datapath='data/aligned_test_face_src_images')

                    #encode faces with vae
                    faces_encoded = encode_faces(faces_2_encode)

                    #encode mrcs
                    encode_faces_mrcs(faces_encoded,ts)
                    print (len(faces_encoded), 'faces are encoded succesfully. Going ahead.')
                    
                    #regenerate faces with vae
                    faces_decoded = decode_faces(faces_encoded)
                    index=0
                    #checking all the detected faces if they are frontal
                    for face_decoded in faces_decoded:
                        cv2.imwrite('data/regenerated_test_face_src_images'+'/'+ts+'_'+str(index)+'.png',(face_decoded* 255.0).astype(int))
                        index=index+1
                    print (len(faces_encoded), 'faces are regenerated succesfully. Going ahead.')
                    
                else:
                    print ('Faces are not detected. Going ahead.')
            
            #quitting with ESC or Q button
            
            if pressed_key == ord('q') or pressed_key == 27:
                print('Quit the application by user command.')
                break
            
        #releasing camera and closeing windows
        camera.release()
        cv2.destroyAllWindows()
        #except:
           #print('Exception while processing the webcam. Quit the application.')
        
        

    


    
