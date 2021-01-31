# -*- coding: utf-8 -*-
"""

@author: yrame
"""
import keras
import os
import numpy as np
from keras.layers import *
from keras import backend as K
from matplotlib import pyplot as plt


# Rest of this cell uses the loss adding method from https://www.kaggle.com/rvislaywade/visualizing-mnist-using-a-variational-autoencoder
def combine_mu_and_sigma(args):
    z_mu, z_log_sigma = args
    epsilon = K.random_normal((K.shape(z_mu)[0], 30),)#numdim
    print(epsilon)
    return z_mu + K.exp(z_log_sigma) * epsilon


def encode_face(face_image):
    
    
    #loading encoder model
    encoder = keras.models.load_model('models/30encoder_192_192.hdpy')
    encoder.summary()
    
    
    print(encoder.input.shape) 
    
    face_image = np.array([face_image/255.0])  # Convert single image to a batch.
    
    #prediction
    
    face_representation = combine_mu_and_sigma(encoder.predict(face_image)).numpy()
    #face_representation = encoder.predict(face_image)
    print('prediction ',
          #face_representation.shape,
          face_representation, 'max value', np.amax(face_representation),'min value', np.amin(face_representation))
    
    return face_representation

    pass

def decode_face(face_representation):
    
    #loading encoder model
    decoder = keras.models.load_model('models/30decoder_192_192.hdpy')
    decoder.summary()
    
    x_mean = np.load('means.npy',)
    x_stds = np.load('stds.npy', )
    e = np.load('evals.npy', )
    v = np.load('evecs.npy', )

    vector = np.array(np.random.uniform(-2, 2))
    vector = face_representation
    vector*=6.1#6.1
    vector = x_mean + np.dot(v, (vector * e).T).T
    np.clip(vector, -2, 2, vector)
    
    print("vector",vector)
    print("face repr",face_representation)
    
    #face_image = np.array([face_image/255.0])  # Convert single image to a batch.
    
    #prediction
    
    face_image = (decoder.predict(face_representation))
    #face_image = (decoder.predict(vector.reshape(1,30)))
    #face_representation = encoder.predict(face_image)
    
    return face_image

    pass

def test_encoder():
    
    #loading test image
    imsize = (192, 192, 3) # Image dimensions
    
    image_path = 'test_face_src_images/1.jpg'
    
    #loading Image to PIL format
    face_image = keras.preprocessing.image.load_img(image_path, 
                                       color_mode="rgb", 
                                       target_size=imsize, 
                                       interpolation="nearest")
    print(type(face_image))

    #transforming image to numpy array
    face_image = keras.preprocessing.image.img_to_array(face_image)
    print(type(face_image), face_image.shape)
    
    print(face_image)

    plt.imshow(face_image.astype(np.uint), interpolation='nearest')
    plt.show()
    
    face_representation = encode_face(face_image)
    
    
    decoded_face_image = decode_face(face_representation)
    plt.imshow((np.squeeze(decoded_face_image, axis=0)*255).astype(np.uint), interpolation='nearest')
    plt.show()
    
    
    #print prediction
    
    pass




if __name__ == '__main__':
    test_encoder()
    #encoder_model()