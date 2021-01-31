# -*- coding: utf-8 -*-
"""

Funcrions for processing frames to detect and decode barcodes
@author: iurii

"""

import cv2
from pyzbar import pyzbar
import numpy as np
import math


#import utils function
from utils import crop_any_rect
from utils import distance_2_pts


def rearrange_qr_corners(p1, p2, p3, p4, x_zero, y_zero):
    """
    Stupid function to properly reorder the points given by qr detector.
    Generally just rearranges the first and the fourth points if needed

    Parameters
    ----------
    p1 : tuple (x,y) first point; 
    p2 : tuple (x,y) second point;   
    p3 : tuple (x,y) third point;   
    p4 : third (x,y) fourth point; 
    x_zero : float min x left qr border point; 
    y_zero : float min y left qr border point; 

    Returns
    -------
    p1_new : tuple (x,y) first out point; 
    p2_new : tuple (x,y) second out point; 
    p3_new : tuple (x,y) third out point; 
    p4_new : tuple (x,y) fourth out point; 

    """
    
    p1_new, p2_new, p3_new, p4_new = p1, p2, p3, p4
    
    p1_zero = distance_2_pts(p1,(x_zero, y_zero))
    #p2_zero = distance_2_pts(p2,(x_zero, y_zero))
    #p3_zero = distance_2_pts(p3,(x_zero, y_zero))
    p4_zero = distance_2_pts(p4,(x_zero, y_zero))
    
    if(p1_zero>p4_zero):
        p1_new, p2_new, p3_new, p4_new = p4, p1, p2, p3
    
    return (p1_new, p2_new, p3_new, p4_new)


def read_barcodes(frame, if_debug=False):
    
    barcodes = pyzbar.decode(frame)
    qr_codes=[]
    qr_codes_corners=[]
    qr_codes_positions=[]
    qr_codes_sizes = []
    messages=[]
    
    for barcode in barcodes:
        x, y , w, h = barcode.rect
        p1, p2 , p3, p4 = barcode.polygon
        #1
        barcode_info = barcode.data.decode('utf-8')
        
        qr_codes.append(crop_any_rect(frame,(x,y,w,h)))
        messages.append(barcode_info)
        
        #rearrange points
        p1, p2 , p3, p4 = rearrange_qr_corners(p1, p2, p3, p4, x, y)
        
        qr_codes_corners.append((p1, p2 , p3, p4))
        
        qr_codes_sizes.append(max(distance_2_pts(p1, p3),distance_2_pts(p2, p4)))
        
        #in debug mode draw some stuff
        if if_debug:
            #cv2.rectangle(frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame,p1,2,(255,0,0))
            cv2.circle(frame,p2,2,(0,255,0))
            cv2.circle(frame,p3,2,(0,0,255))
            cv2.circle(frame,p4,2,(255,255,0))
            #print(p1, p2 , p3, p4)
            #2
            cv2.putText(frame, barcode_info, (x + 6, y - 6), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 255), 1)
            
            #write decoded results to file 
            with open("QRcode_result.txt", mode ='w') as file:
                file.write("Recognized Barcode:" + barcode_info)

        
    return frame, qr_codes, qr_codes_positions,qr_codes_corners,qr_codes_sizes, messages

def get_qr_code_transformation(im, qr_code_points, targer_qr_size=400):

    size = im.shape
    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
                                qr_code_points[0],     # Nose tip
                                qr_code_points[1],     # Chin
                                qr_code_points[2],     # Left eye left corner
                                qr_code_points[3],     # Right eye right corne
                            ], dtype="double")
     
    # 3D model points.
    model_points = np.array([
                                (-targer_qr_size/2, -targer_qr_size/2, 0.0),             
                                (-targer_qr_size/2, targer_qr_size/2, 0),      
                                (targer_qr_size/2, targer_qr_size/2, 0),     
                                (targer_qr_size/2, -targer_qr_size/2, 0),     
                            ])
    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                              [[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype = "double"
                              )
    
    #print ("Camera Matrix :\n {0}".format(camera_matrix))
    
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    # print ("Rotation Vector:\n {0}".format(rotation_vector))
    # print ("Translation Vector:\n {0}".format(translation_vector))
    rotation = (math.degrees(rotation_vector[0]),math.degrees(rotation_vector[1]),math.degrees(rotation_vector[2]))
    translation =(translation_vector[0]/targer_qr_size,translation_vector[1]/targer_qr_size,translation_vector[2]/targer_qr_size)
    
    #print ("Translation Vector deg:", rotation)
    #print ("Rotation Vector deg:", translation)
    return(rotation, translation)


def handle_mrc_frame(frame, if_debug=False):
    
    processed_frame,qr_codes, qr_codes_positions, qr_codes_corners, qr_codes_sizes, messages = read_barcodes(frame, if_debug=False)
    rotations, translations = [],[]
    for qr_code_corners in qr_codes_corners:
        rotation, translation = get_qr_code_transformation(processed_frame, qr_code_corners)
        rotations.append(rotation)
        translations.append(translation)
        
    return(processed_frame,qr_codes, qr_codes_positions, qr_codes_corners, qr_codes_sizes, messages, rotations, translations)

def test_stream():
    #Load video stream from default webcam
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    #analyzing frames
    while ret:
        ret, frame = camera.read()
        
        processed_frame,qr_codes, qr_codes_positions, qr_codes_corners, qr_codes_sizes, messages, rotation, translation = handle_mrc_frame(frame, True)
        
        if(len(qr_codes)>0):
            cv2.imshow('first detected QR code', qr_codes[0])  
            # print ("Rotation Vector deg:", rotation)
            # print ("Translation Vector deg:", translation)
        
        #output messages
        for message in messages:    
            print(message)
            pass
        cv2.imshow('Original frame', frame)
        cv2.imshow('Processed frame', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    #releasing camera and closeing windows
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_stream()