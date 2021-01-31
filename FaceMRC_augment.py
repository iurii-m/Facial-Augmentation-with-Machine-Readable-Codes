# -*- coding: utf-8 -*-
"""
Console application to create MRC (QR Code) from the input image or webcam capture.
based on the example of using the PyWavefront module

@author: iurii
"""

import ctypes
import os
import sys
sys.path.append('..')

import pyglet
from pyglet.gl import *

from pywavefront import visualization
from pywavefront import Wavefront

import numpy as np
import cv2

from webcam import Webcam

from pyzbar import pyzbar

from decode_face_mrc import handle_mrc_frame

from utils import overlay_image_alpha

from base64_to_array import array_2_base64
from base64_to_array import base64_2_array

from vae_encode_face import decode_faces

from generate_3D_face_obj import generate_3D_face_obj

import time


from threading import Thread


#current name IDs of models to show
current_name_ids = []

# Create absolute path from this module
file_abspath = [os.path.join(os.path.dirname(__file__), 'data/1_obj.obj')]

rotation = 0.0

#faces_3D = [Wavefront(file_abspath[0])]
faces_3D=[]
faces_to_decode = []


#face size multiply coefficient
face_mult = 300

#pyglet scene and window parameters
window_x = 400
window_y = 400
window = pyglet.window.Window(window_x, window_y, caption='Face Model', resizable=False, visible=True)

lightfv = ctypes.c_float * 4


#util parameters
result_resize_coeff = 1.5
latent_size = 64

event_loop = pyglet.app.EventLoop()

webcam = Webcam()
webcam_thread = webcam.start()


#read already proceeded qr messages
saved_QR_messages_file = open("data/generated_3D_models/saved_QR_messages.txt", "r")
saved_QR_messages = saved_QR_messages_file.readlines()
# print(saved_QR_messages)
# print(saved_QR_messages[0])
# print('________', saved_QR_messages[0].split(' ')[0],'_____',saved_QR_messages[0].split(' ')[1])
saved_QR_messages_file.close()
  


def face_3d_model_handler(faces_to_decode, ts):
    
    faces_decoded = decode_faces(faces_to_decode)
    print("thread is started")
    for ind in range(len(faces_decoded)):
        #generate 3D Model
        generate_3D_face_obj((faces_decoded[ind]*255.0).astype(int),ts+'_'+str(ind),'data/generated_3D_models/')
        cv2.imwrite('data/decoded_face_images'+'/'+ts+'_'+str(ind)+'.png',(faces_decoded[ind]*255.0).astype(int))
    #print("thread is finished")


thread_face_3d_loading = Thread(target = face_3d_model_handler, args = (faces_to_decode,'_'))


@event_loop.event
def on_window_close(window):
    event_loop.exit()

@window.event
def on_resize(width, height):
    viewport_width, viewport_height = window.get_framebuffer_size()
    glViewport(0, 0, viewport_width, viewport_height)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(40.0, float(width) / height, 1.0, 100.0)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_MODELVIEW)
    return True

@window.event
def on_draw():
    
    global current_name_ids
    global faces_3D
    global thread_face_3d_loading
    global faces_to_decode
    
    window.clear()
    
    #global webcam    
    #cv2.imshow('Source Webcam stream', webcam.get_current_frame())
    
    #get current frame and process it
    frame = webcam.get_current_frame()
    result = frame
    qr_codes, qr_codes_positions, qr_codes_corners, qr_codes_sizes, messages, rotations, translations = [],[],[],[],[],[],[]
    try:
        processed_frame,qr_codes, qr_codes_positions, qr_codes_corners, qr_codes_sizes, messages, rotations, translations = handle_mrc_frame(frame)
    except:
        pass
        
    
    enc_faces = []
    enc_qr_corners = []
    enc_qr_sizes = []
    enc_messages = []
    enc_rotations = []
    enc_translations = []
    
    
    #checking QR code messages 
    for ind in range(len(qr_codes)):
        
        #decode latent face representation   
        try:
            face_encoded = base64_2_array(messages[ind].encode(),np.float16)
        except:
            #print("decoding exception")
            continue
        
        #collect messages if size is correct        
        if face_encoded.shape==(latent_size,):
            enc_faces.append(face_encoded)
            enc_qr_corners.append(qr_codes_corners[ind])
            enc_qr_sizes.append(qr_codes_sizes[ind])
            enc_messages.append(messages[ind])
            enc_rotations.append(rotations[ind])
            enc_translations.append(translations[ind])
            
            
    #generate data for demonstration (face image and 3D model), except messages, that were already proceeded before

    dem_name_ids = []
    dem_qr_corners = []
    dem_qr_sizes = []
    dem_rotations = []
    dem_translations = []
    
    faces_to_decode = []
    
    #timestamp
    ts = str(time.time())
         
    counter=0
    for ind in range(len(enc_faces)):
        if_decoded = False
        #check if message is already decoded
        for ind_2 in range(len(saved_QR_messages)):
            
            #if message is already decoded then store its name 
            #print(messages[ind] in saved_QR_messages[ind_2])
            if (messages[ind] in saved_QR_messages[ind_2]): 
            #if messages[ind]==saved_QR_messages[ind_2].split(' ')[1]:
                #print("already decoded")
                if_decoded=True
                
                dem_name_ids.append(saved_QR_messages[ind_2].split(' ')[0])
  
                break
        #if not yet decoded thes append to the list of faces to decode and assign new name which is appended to the 
        #saved_QR_messages valiable and file with the corresponding message
        if not if_decoded:
            faces_to_decode.append(enc_faces[ind])
            dem_name_ids.append(ts+'_'+str(counter))
            
            saved_QR_messages.append(ts+'_'+str(counter)+' '+enc_messages[ind])
            
            saved_QR_messages_file = open("data/generated_3D_models/saved_QR_messages.txt", "a+")
            saved_QR_messages_file.write(ts+'_'+str(counter)+' '+enc_messages[ind])
            saved_QR_messages_file.write("\n")
            saved_QR_messages_file.close()
            counter = counter+1
            
        
        #in any case store demonstration parameters
        dem_qr_corners.append(enc_qr_corners[ind])
        dem_qr_sizes.append(enc_qr_sizes[ind])
        dem_rotations.append(enc_rotations[ind])
        dem_translations.append(enc_translations[ind])
            
    
    #generate face images with vae, save them and create 3D face models
    if(len(faces_to_decode)>0) & (not thread_face_3d_loading.is_alive()):
        #print("to start frame")
        thread_face_3d_loading = Thread(target = face_3d_model_handler, args = (faces_to_decode,ts))
        thread_face_3d_loading.start()
        
        # faces_decoded = decode_faces(faces_to_decode)
        # for ind in range(len(faces_decoded)):
        #     #generate 3D Model
        #     generate_3D_face_obj((faces_decoded[ind]*255.0).astype(int),ts+'_'+str(ind),'data/generated_3D_models/')
        #     cv2.imwrite('data/decoded_face_images'+'/'+ts+'_'+str(ind)+'.png',(faces_decoded[ind]*255.0).astype(int))
            
    
    #reassing the models if required
    #check if name lists are the same
    #TODO check lists by elements and size to avoid reloading after changing order of the name_IDs in the list
    
    #print('before start the draw', len(dem_name_ids)>0 ,not thread_face_3d_loading.is_alive())
    #check zero dem_name_ids - in that case skip the 3d part entirely
    if (len(dem_name_ids)>0) & (not thread_face_3d_loading.is_alive()):
        
        #sorting all the data according to the names
        #dem_name_ids,dem_qr_corners,dem_qr_sizes,dem_rotations,dem_translations=[d for d in sorted(zip(dem_name_ids,dem_qr_corners,dem_qr_sizes,dem_rotations,dem_translations))]
        dem_qr_corners = [x for _,x in sorted(zip(dem_name_ids,dem_qr_corners))]
        dem_qr_sizes = [x for _,x in sorted(zip(dem_name_ids,dem_qr_sizes))]
        dem_rotations = [x for _,x in sorted(zip(dem_name_ids,dem_rotations))]
        dem_translations = [x for _,x in sorted(zip(dem_name_ids,dem_translations))]
        dem_name_ids.sort()
        
        #if they are not the same then reload models. Recent sort is applyed to avoid difference in order at this point
        if not (current_name_ids==dem_name_ids):
            current_name_ids = dem_name_ids
            reload_faces_3D_models(dem_name_ids, 'data/generated_3D_models/')
             
        
            
        #processing QR code message to extract 
        for ind in range(len(faces_3D)):   
           #go ahead if MRC is ok and the face image was decoded
           
           #preparing window for visualisation
           window.clear()
       
           
           #Exctacting rotation
           x_rot, y_rot, z_rot = dem_rotations[ind]
           
           #Exctacting translation V1
           # x_tr, y_tr, z_tr = dem_translations[ind]
           # tr_mult = 1.0/dem_qr_sizes[ind]
           # x_tr, y_tr, z_tr =x_tr/tr_mult, y_tr/tr_mult, z_tr/tr_mult
           
           #Exctacting translation V2
           x_tr = (dem_qr_corners[ind][0][0]+dem_qr_corners[ind][1][0]+dem_qr_corners[ind][2][0]+dem_qr_corners[ind][3][0])/4
           y_tr = (dem_qr_corners[ind][0][1]+dem_qr_corners[ind][1][1]+dem_qr_corners[ind][2][1]+dem_qr_corners[ind][3][1])/4
           
           
           #Excracting size coefficient
           qr_size = dem_qr_sizes[ind]
    
           #render 3d face model
           draw_face_3D(faces_3D[ind],x_rot, y_rot, z_rot)

           #transforming rendered image to array
           image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
       
           im_arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
           im_arr = im_arr.reshape(window_x, window_y, 4)
           im_arr = im_arr[::-1,:,0:3]
           im_arr = cv2.cvtColor(im_arr, cv2.COLOR_RGB2BGR)
           #cv2.imshow('image',im_arr)
           
    
           #Combine images
           #resize image with face renrer      
           face_render = cv2.resize(im_arr, (max(1,int(im_arr.shape[1]/face_mult*qr_size)),max(1,int(im_arr.shape[0]/face_mult*qr_size))))
           #thresholding to extract mask
           _,face_render_mask = cv2.threshold(cv2.cvtColor(face_render, cv2.COLOR_RGB2GRAY),10,255,cv2.THRESH_BINARY)
         
           #overlay images
           result = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
           face_render_alpha = cv2.cvtColor(face_render, cv2.COLOR_RGB2RGBA).copy()
           result = overlay_image_alpha(result,face_render_alpha,
                                       (frame.shape[1]/2*0-face_render_alpha.shape[1]/2+int(x_tr), frame.shape[0]/2*0-face_render_alpha.shape[0]/2+int(y_tr)),
                                       face_render_mask/255.0)
           result = cv2.cvtColor(result, cv2.COLOR_RGBA2RGB)
            
    cv2.imshow('result', cv2.resize(result, (int(result.shape[1]*result_resize_coeff) , int(result.shape[0]*result_resize_coeff))))
  
def draw_face_3D(face_3D_model, x_rot, y_rot, z_rot):
    """
    Parameters
    ----------
    face_3D_model : TYPE
        DESCRIPTION.
    x_rot : TYPE
        DESCRIPTION.
    y_rot : TYPE
        DESCRIPTION.
    z_rot : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    glLoadIdentity()
    #rotate face 
    glTranslated(0, 0, -5)
    glRotatef(x_rot, 1.0, 0.0, 0.0)
    #glRotatef(rotation, 0.0, 1.0, 0.0)
    glRotatef(-y_rot, 0.0, 1.0, 0.0)
    glRotatef(- z_rot, 0.0, 0.0, 1.0)
    
    #visualize
    visualization.draw(face_3D_model)




def reload_faces_3D_models(name_ids, modelspath):
    global faces_3D
    global if_faces_3d_loaded
    
    faces_3D = []
    
    for name_id in name_ids:
        face_3D = Wavefront(modelspath +'/'+ name_id + '.obj')
        faces_3D.append(face_3D)
        

def update(dt):
    global rotation
    rotation += 90.0 * dt

    if rotation > 720.0:
        rotation = 0.0


pyglet.clock.schedule(update)
pyglet.app.run()


print("here")


