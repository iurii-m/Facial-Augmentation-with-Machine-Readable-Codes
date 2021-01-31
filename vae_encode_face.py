# -*- coding: utf-8 -*-
"""

Main VAE Functionality train and endode/decode. 

"""


import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES']="0"

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

K = tf.keras.backend


from scheduler import *
import numpy as np
import matplotlib.pyplot as plt
import argparse

import vae_generator
import cv2

import vae_face_utils

import base64_to_array

# import dlib

# network parameters
batch_size = 32
epochs = 10
image_size = 224
input_shape = (image_size, image_size, 3)
latent_dim = 96
id = '10'
max_samples = 0
split = 0.8


# predictor_path = 'models/shape_predictor_68_face_landmarks.dat'
# face_rec_model_path = 'models/dlib_face_recognition_resnet_model_v1.dat'



# detector = dlib.get_frontal_face_detector()
# sp = dlib.shape_predictor(predictor_path)
# facerec = dlib.face_recognition_model_v1(face_rec_model_path)
# # Set tolerance for face detection smaller means more tolerance for example -0.5 compared with 0
# tolerance = 0

def make_graph(layers):
    result = layers[0]
    for i in range(1, len(layers)):
        result = layers[i](result)

    return result


fr_model = tf.keras.models.load_model("models/mobileNetV2_trained_test.hdf5")
fr_model.summary()
fr_model.trainable = False
fr_model.summary()
# print(model.trainable)

def face_rec_loss(input1, input2):
    
    model = fr_model
    print(input1.shape)
    # input1 = tf.image.resize(input1, (224,224))
    # input2 = tf.image.resize(input2, (224,224))
    # print(input1.shape)

    output1 = model(input1)
    output2 = model(input2)
    fl_output1 = K.flatten(output1)
    fl_output2 = K.flatten(output2)
    
    # dot_prod = tf.tensordot(fl_output1, fl_output2, 1)
    similarity = tf.keras.losses.binary_crossentropy(fl_output1, fl_output2)
    
    
    # similarity = tf.keras.layers.Dot(axes=1)([output1, output2])


    return similarity
    

def make_models():

    # reparameterization trick
    # instead of sampling from Q(z|X), sample epsilon = N(0,I)
    # z = z_mean + sqrt(var) * epsilon
    def sampling(z_mean_z_log_var):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments
            z_mean_z_log_var (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = z_mean_z_log_var
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    # VAE model = encoder + decoder

    #------------ENC------------

    # build encoder model
    main_inputs = tf.keras.layers.Input(shape=input_shape, name='encoder_input')  # image_sizeximage_sizex3
    encoder_graph = make_graph([
        main_inputs,
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="SAME", activation='relu'),  # 64x64x32
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="SAME", activation='relu'),  # 32x32x64
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="SAME", activation='relu'),  # 16x16x64
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="SAME", activation='relu'),  # 8x8x64
        tf.keras.layers.Flatten(),  # 4096
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(512, activation='relu')
        
    ])
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(encoder_graph)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(encoder_graph)

    # use reparameterization trick to push the sampling out as input
    z = tf.keras.layers.Lambda(sampling, name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder_model = tf.keras.models.Model(main_inputs, [z, z_mean, z_log_var], name='encoder')
    encoder_model.summary()
    tf.keras.utils.plot_model(encoder_model, to_file='vae_mlp_encoder.png', show_shapes=True)

    #------------DEC------------

    # build decoder model
    latent_inputs = tf.keras.layers.Input(shape=(latent_dim, ), name='z_sampling')
    decoder_graph = make_graph([
        latent_inputs,
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(25088, activation='relu'),
        tf.keras.layers.Reshape(target_shape=(14, 14, 128)),
        tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding="SAME", activation='relu'),
        tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding="SAME", activation='relu'),
        tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="SAME", activation='relu'),
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="SAME", activation='relu'),
        tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding="SAME")
    ])

    # instantiate decoder model
    decoder_model = tf.keras.models.Model(latent_inputs, decoder_graph, name='decoder')
    decoder_model.summary()
    tf.keras.utils.plot_model(decoder_model, to_file='vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    main_outputs = decoder_model(encoder_model(main_inputs)[0])  # 0 refers to Z in encoder model outputs
    vae = tf.keras.models.Model(main_inputs, main_outputs, name='vae_mlp')

    # Make loss function
    # VAE loss = mse_loss or xent_loss + kl_loss
    
    # if args.mse:
    #     print("TUTA TRUE")
    #     reconstruction_loss = tf.keras.losses.mse(K.flatten(main_inputs), K.flatten(main_outputs))
    # else:
    #     print("TUTA false")
    #     reconstruction_loss = tf.keras.losses.binary_crossentropy(K.flatten(main_inputs), K.flatten(main_outputs))
    
    reconstruction_loss = tf.keras.losses.binary_crossentropy(K.flatten(main_inputs), K.flatten(main_outputs))
    # recognition_loss =face_rec_loss(main_inputs,main_outputs)
    
    fr_loss = face_rec_loss(main_inputs, main_outputs)
    
    #print("shapes", main_inputs.shape,main_outputs.shape)
    #perception_loss

    reconstruction_loss *= image_size * image_size
    
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    
    print("shapes", reconstruction_loss.shape, kl_loss.shape,  fr_loss.shape)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    

    fr_loss *= image_size * image_size
    
    vae_loss = K.mean( reconstruction_loss + kl_loss + fr_loss)#*recognition_loss 
    
    print("shapes", reconstruction_loss.shape, kl_loss.shape, vae_loss.shape, fr_loss.shape)
    
    vae.add_loss(vae_loss)
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    vae.compile(optimizer=optimizer)
    
    #vae.compile(optimizer='adam')
    
    #vae.compile(optimizer='adam', loss=vae_loss)
    #vae.summary()
    tf.keras.utils.plot_model(vae, to_file='vae_mlp.png', show_shapes=True)

    return vae, encoder_model, decoder_model


def encode_faces(faces):

    vae, encoder, decoder = make_models()

    save_file_name = os.path.join('models', id, f'vae_10_64_100_0.0_256')
    save_file_name = save_file_name + '.tf'
    vae.load_weights(save_file_name)

    
    encoded_faces = []

    for face_im in faces:
        #TODO - change here
        #face detect optional
        _, face_im = vae_face_utils.process_frame(face_im)
        face_im = cv2.resize(face_im, (image_size,image_size), interpolation=cv2.INTER_CUBIC)
  
        
        encoded = encoder.predict((face_im / 255.0).reshape(1, image_size, image_size, 3))[0]
        encoded_faces.append(encoded[0].astype('float16'))

    return(encoded_faces)

def decode_faces(encoded_faces):
    vae, encoder, decoder = make_models()
    save_file_name = os.path.join('models', id, f'vae_10_64_100_0.0_256')
    save_file_name = save_file_name + '.tf'
    vae.load_weights(save_file_name)
    
    faces = []
    
    for encoded_face in encoded_faces:
        face = decoder.predict(encoded_face.reshape(1, 64))[0]        
        faces.append(face)
    
    return(faces)


if __name__ == '__main__':
    save_file_name = os.path.join('models', id, f'vae_10_64_100_0.0_256')
    save_file_path = save_file_name + '.tf'
    parser = argparse.ArgumentParser()

    parser.add_argument("-w", "--weights", help="Load h5 or tf model trained weights", default=save_file_path)
    parser.add_argument('-f', '--file', help='File to predict. Works only when -w is used.')#, default="2.png")
    parser.add_argument('-c', '--compress', help='File to compress. Works only when -w is used.')
    parser.add_argument('-d', '--decompress', help='File to decompress. Works only when -w is used.')
    parser.add_argument('-F',
                        '--force',
                        help='force image to enter the network, regardless of the face preprocessing',
                        action='store_true')
    parser.add_argument('-D',
                        '--double',
                        help='To compress or decompress using float32 instead of float16.',
                        action='store_true')
    parser.add_argument("-m",
                        "--mse",
                        help="Use mse loss instead of binary cross entropy (default)",
                        action='store_true')
    args = parser.parse_args()

    vae, encoder, decoder = make_models()
    print('looking for', args.weights)
    if args.weights and os.path.isfile(args.weights + '.index') and False:
        print('found')
        vae.load_weights(args.weights)
        if args.file:
            if args.force:
                img = cv2.resize(cv2.imread(args.file), (image_size, image_size))
            else:
                img = vae_face_utils.get_face(args.file, resize=(image_size, image_size))

            cv2.imshow('compressed image', img)
            encoded = encoder.predict((img / 255).reshape(1, image_size, image_size, 3))[0]
            print('encoded',encoded)
            with open(args.file + '.compressed', 'wb') as f:
                if not args.double:
                    encoded[0].astype('float16').tofile(f)
                    
                                        
                    print(base64_to_array.array_2_base64(encoded[0].astype('float16')))
                    
                    
                else:
                    encoded[0].tofile(f)

            #encoded = np.linspace(1, -1, 64).reshape(1, 64)
            prediction = decoder.predict(encoded)
            prediction = prediction[0]
            cv2.imshow('result', prediction)
            cv2.imwrite(args.file + '.compressed.png', (prediction * 255).astype(int))

        elif args.compress:
            if args.force:
                img = cv2.resize(cv2.imread(args.file), (image_size, image_size))
            else:
                
                img = vae_face_utils.get_face(args.file, resize=(image_size, image_size))

            cv2.imshow('compressed image', img)
            encoded = encoder.predict((img / 255).reshape(1, image_size, image_size, 3))[0]
            print(encoded)
            with open(args.compress + '.compressed', 'wb') as f:
                if not args.double:
                    encoded[0].astype('float16').tofile(f)
                    
                else:
                    encoded[0].tofile(f)

        elif args.decompress:

            with open(args.decompress, 'rb') as f:
                if not args.double:
                    encoded = np.fromfile(f, dtype='float16')
                else:
                    encoded = np.fromfile(f, dtype='float32')

            #encoded = np.linspace(1, -1, 64).reshape(1, 64)
            prediction = decoder.predict(encoded.reshape(1, 64))
            prediction = prediction[0]
            cv2.imshow('decompressed image', prediction)
            cv2.imwrite(args.decompress + '.png', (prediction * 255).astype(int))

        cv2.waitKey()

    else:
        
        # img1 = cv2.resize(cv2.imread('data/test_face_src_images/1.jpg'), (image_size, image_size))
        # img2 = cv2.resize(cv2.imread('data/test_face_src_images/6.jpg'), (image_size, image_size))
        # #img2 = cv2.resize(cv2.imread('data/test_face_src_images/noface.png'), (image_size, image_size))
        # face_rec_loss(img1, img2)
        

        
        print('not found')
        # train the autoencoder
        #dataset_path = 'E:/DataZoo/archive/192_192/total/images'
        
        #dataset_path = '/media/bigdrive/Iurii/cg3d/img_align_celeba_frontal/img_align_celeba'
        dataset_path = '/media/bigdrive/Iurii/cg3d/images'
        train_generator = vae_generator.DataGenerator(dataset_path, batch_size, test=False, split=split, max_samples=max_samples)
        test_generator = vae_generator.DataGenerator(dataset_path, batch_size, test=True, split=split, max_samples=max_samples)
        ckpt = tf.keras.callbacks.ModelCheckpoint(save_file_name + '.tf',
                                                  verbose=1,
                                                  monitor='val_loss',
                                                  save_best_only=True,
                                                  save_weights_only=False)
        
        cosan = CosineAnnealingScheduler(T_max=epochs, eta_max=0.001, eta_min=0.00001, verbose=1)
        
        vae.fit(train_generator,
                validation_data=test_generator,
                epochs=epochs,
                shuffle=True,
                callbacks=[ckpt,cosan],
                max_queue_size=30,
                workers=8,
                use_multiprocessing=True)
        vae.save_weights(save_file_name)
