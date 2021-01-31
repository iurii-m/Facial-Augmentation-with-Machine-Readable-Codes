# -*- coding: utf-8 -*-
"""
Simple face alignment by keypoints from mtcnn detection
Returns rectangular region  

@author: iurii
"""

import cv2
from mtcnn import MTCNN
import numpy as np
from utils import crop_face
import math

def _find_center_pt(keypoints):
    #find central point of the face landmarks
    x = 0
    y = 0
    num = len(keypoints)
    for pt in keypoints:        
        x += keypoints[pt][0]
        y += keypoints[pt][1]
        
    x //= num
    y //= num
    return (x,y)

def _find_face_dim(bounding_box):
    #find dimention of face in pixels as average of width and height
    dim = (bounding_box[2]+ bounding_box[3])/2
    return dim

def _find_face_dim_keypoints(keypoints):
    #find dimention of face in pixels as average of width and height
    left_eye = (keypoints['left_eye'])
    right_eye = (keypoints['right_eye'])
    nose = (keypoints['nose'])
    mouth_left = (keypoints['mouth_left'])
    mouth_right = (keypoints['mouth_right'])
    
    center_of_face = _find_center_pt(keypoints) 
    
    left_eye_2_center = distance_2_pts(left_eye,center_of_face)
    right_eye_2_center = distance_2_pts(right_eye,center_of_face)
    nose_2_center = distance_2_pts(nose,center_of_face)
    mouth_left_2_center = distance_2_pts(mouth_left,center_of_face)
    mouth_right_2_center = distance_2_pts(mouth_right,center_of_face)
    
    distances = [left_eye_2_center,
                 right_eye_2_center,
                 nose_2_center,
                 mouth_left_2_center,
                 mouth_right_2_center]
    
    
    return max(distances)

def _angle_between_2_pt(p1, p2):
    # to calculate the angle rad by two points   
    x1, y1 = p1
    x2, y2 = p2
    delta_x = (x2 - x1)
    #checking zero
    if (delta_x == 0):
        delta_x = 0.00001
    tan_angle = (y2 - y1) / (delta_x)
    return (np.degrees(np.arctan(tan_angle)))

def _get_rotation_matrix(keypoints, rotation_point, face_img, scale):
    # get rotation matrix as average angle between eyes points and mouse corner points
    #face points
    left_eye_pt = keypoints['left_eye']
    right_eye_pt = keypoints['right_eye']
    left_mouth_pt = keypoints['mouth_left']
    right_mouth_pt =keypoints['mouth_right']
    #angles
    eye_angle = _angle_between_2_pt(left_eye_pt, right_eye_pt)
    mouth_angle  = _angle_between_2_pt(left_mouth_pt, right_mouth_pt)
    average_angle = (eye_angle + mouth_angle)/2
    #rotation
    M = cv2.getRotationMatrix2D((rotation_point[0], rotation_point[1]), average_angle, scale )
    return M

def distance_2_pts(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)



#simply cropp the central part of the image
def _no_detected_face(cv_image, crop_ratio = 0.7, face_size=(224,224)):
    
    h, w, c = cv_image.shape
    new_bounding_box=[h/2-h*crop_ratio/2, w/2-w*crop_ratio/2, h*crop_ratio, w*crop_ratio]
    
    cropped_image = crop_face(cv_image, new_bounding_box, (w/2,h/2)) 
    
    return cv2.resize(cropped_image, face_size, interpolation=cv2.INTER_CUBIC)   

# return align faces, numberof detected faces,index on the main face
def _align_faces(cv_image,detector, face_scale=1.1, face_size=(224,224)):
     
    print("test31")  
    #check if image is too small, and if so then increase the size
    if(cv_image.shape[0]+cv_image.shape[1]<200):
        print('enlarged')
        cv_image = cv2.resize(cv_image, (cv_image.shape[1]*3,cv_image.shape[0]*3), interpolation=cv2.INTER_CUBIC)
    
    h, w, c = cv_image.shape 
    print("test3")
    #detect faces
    detection = detector.detect_faces(cv_image) 
    del detector
    print("test4")
    #print("face detected ", len(detection))
    output_imgs = list()
    #index of the main face (closest to the center)
    main_idx = 0 
    #current distance from the center. search for minimal
    l_dist = h*w
    
    frontalface_len = 0 
    
    #cycle idx
    idx = 0
    for face in detection:
        #face location
        bounding_box = face['box']
        keypoints = face['keypoints']
        
        left_eye = (keypoints['left_eye'])
        right_eye = (keypoints['right_eye'])
        nose = (keypoints['nose'])
        mouth_left = (keypoints['mouth_left'])
        mouth_right = (keypoints['mouth_right'])
        
        
        #estimating frontality of the face
        rel1 = distance_2_pts(left_eye, nose)/max(0.0001,distance_2_pts(right_eye, nose))
        rel2 = distance_2_pts(mouth_left, nose)/max(0.0001,distance_2_pts(mouth_right, nose))
        thresh = 0.2
        if(abs(1-rel1)>thresh)or(abs(1-rel2)>thresh):
            print("non frontal mtcnn skip")
            continue
            
        #find center of the face
        center_of_face = _find_center_pt(keypoints)   
        #chech distance to the center. if true assign new index  
        center_distance = distance_2_pts((w/2,h/2),center_of_face)
            #print("points ", w,h,center_of_face, center_distance,l_dist)
        if(l_dist > center_distance):
            main_idx = idx
            l_dist = center_distance            
        
        # #find face dimention
        # face_dim =_find_face_dim(bounding_box)*face_scale
        
        
        face_dim =_find_face_dim_keypoints(keypoints)*face_scale
        
        
        
        #estimating new bounding box
        new_bounding_box=[center_of_face[0]-face_dim/2,center_of_face[1]-face_dim/2, face_dim, face_dim]
   
        #get rotation angle
        trotate = _get_rotation_matrix(keypoints, center_of_face, cv_image, scale=1)
        #2d transformation
        warped = cv2.warpAffine(cv_image, trotate, (w, h))
        #cropp face to proper square, resize to required dimentions, and output          
        cropped_image = crop_face(warped , new_bounding_box, center_of_face, True, face_scale)
        #write to the output
        output_imgs.append(cv2.resize(cropped_image, face_size, interpolation=cv2.INTER_CUBIC))
        idx = idx+1
        frontalface_len = frontalface_len +1
        
    #if no faces detected ruth the spacial handler_method
    if frontalface_len<1:
          print("empty image ")
          output_imgs.append(_no_detected_face(cv_image, crop_ratio = 0.7, face_size=face_size))
         
    return output_imgs, frontalface_len ,main_idx

def _read_image_2_cvMat(imagepath):
    image = cv2.cvtColor(cv2.imread(imagepath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return image






