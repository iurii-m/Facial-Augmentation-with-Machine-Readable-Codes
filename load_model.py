# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:01:46 2021

@author: ana

This program loads a model ()

"""

import numpy as np

import tensorflow.keras
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LearningRateScheduler

import skimage.transform
import skimage.io

def crop_center(img,crop_persentage):
    x,y,c = img.shape
    wdth_new = int(x*max(0,min(crop_persentage,1)))
    hght_new = int(y*max(0,min(crop_persentage,1)))
    startx = (x-wdth_new)//2
    starty = (y-hght_new)//2   
    return img[startx:startx+wdth_new,starty:starty+hght_new]

def image_preprocessed(dataset_path = "", 
                       file_name = "",
                       im_size = (224,224,3), 
                       norm_val = 1, 
                       cropp_percentage = 0.8, 
                       sub_mean =[0.5,0.5,0.5]):
    
    return ((skimage.transform.resize(crop_center(skimage.io.imread(
           dataset_path + str(file_name)), cropp_percentage), im_size)-sub_mean)/norm_val)

def test_image(filepath = "",
               filename = "",
               modelpath = "",
               cropp_percentage = 0.8,
               sub_mean = [0.5671085880143463,0.4296848511769544,0.36655578598995664] ,
               im_size = (224,224,3)):
    """
    Parameters
    ----------

    Returns
    -------
    prediction as numpy darray.
    """


    model = load_model(modelpath)
    model.summary()

    
    image_array = np.array([skimage.transform.resize(skimage.io.imread(filepath + filename), (224,224,3))])/(1)   
    
    image_array = image_preprocessed(filepath, filename)
    
    image_array = image_array[np.newaxis, ...] #because the model needs to receive an additional dimension
    
    print(image_array)
    print(image_array.shape)
    
    result = model.predict(image_array)
    
    print(result)
    print(result.shape)  # result[0].shape in case of multiple output tensors

    return result

def test_2_images(filepath1 = "",
                  filename1 = "",
                  filepath2 = "",
                  filename2 = "",
                  modelpath = "",
                  cropp_percentage = 0.8,
                  sub_mean = [0.5671085880143463,0.4296848511769544,0.36655578598995664] ,
                  im_size = (224,224,3)):
    """
    Parameters
    ----------

    Returns
    -------
    similarity : float value between 0 and 1
        Similarity between 2 identities, where 1 is high similarity and 0 is low similarity.

    """
    
    model = load_model(modelpath)
    model.summary()
    
    print(model.trainable)
    model.trainable = False
    print(model.trainable)
    
    print(model.input.shape)

    image_array1 = np.array([skimage.transform.resize(skimage.io.imread(filepath1+filename1), (224,224,3))])/(1)   
    image_array1 = image_preprocessed(filepath1, filename1)
    image_array1 = image_array1[np.newaxis, ...]
    
    image_array2 = np.array([skimage.transform.resize(skimage.io.imread(filepath2+filename2), (224,224,3))])/(1)   
    image_array2 = image_preprocessed(filepath2, filename2)
    image_array2 = image_array2[np.newaxis, ...]
    
    # print(image_array1)
    # print(image_array1.shape)
    # result1 = model.predict(image_array1)
    # print(result1)
    # print(result1.shape)
    
    # print(image_array2)
    # print(image_array2.shape)
    # result2 = model.predict(image_array2)
    # print(result2)
    # print(result2.shape)
    
    # a = np.squeeze(np.array(result1, dtype=np.float32))
    # b = np.squeeze(np.array(result2, dtype=np.float32))
    
    # similarity  = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    
    
    # return similarity

if __name__ == '__main__':
    
    #identity 1
    dir1 = "vgg_some_test_images/n000029/"
    file1 = "0001_01.jpg"
    file1_2 = "0008_05.jpg"
    
    #identity 2
    dir2 = "vgg_some_test_images/n000001/"
    file2 = "0003_01.jpg"
    
    modelpath = "mobileNetV2_trained_test.hdf5"
    
    #print(test_image(dir1, file1, modelpath))
    print(test_2_images(dir1, file1, dir2, file2, modelpath))           #different people - prints a low similarity value
    #print(test_2_images(dir1, file1, dir1, file1_2, modelpath))        #same person - prints a high similarity value
    
