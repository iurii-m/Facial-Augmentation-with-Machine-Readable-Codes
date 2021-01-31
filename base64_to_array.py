# -*- coding: utf-8 -*-
"""

Some functions to transform from np.array to compressed base64 string
@author: iurii

"""

import base64
import numpy as np
import zlib
import sys

def array_2_base64(array_to_encode):
    """
    compress and encode to base64

    Parameters
    ----------
    array_to_encode : np.array
        array to be encoded

    Returns
    -------
    message : string 
        result string

    """

    message = base64.b64encode(zlib.compress(array_to_encode,9))
        
    return message

def base64_2_array(message_to_decode, to_type):
    """
    base 64 decoding to byte stream, decompressing, and transforming to the defined type 

    Parameters
    ----------
    message_to_decode : string
        message to be decoded

    Returns
    -------
    message : numpy.type
        type of numpy array

    """
       
    decoded_array = np.frombuffer(zlib.decompress(base64.decodebytes(message_to_decode)), dtype=to_type)

    return decoded_array


def test_base64_generation_from_array():
    """
    basic test with generated array
    """
    
    #generate random array
    random_array=np.random.uniform(0, 1.0, size=(128,))
    
    #transforming range and type to some int    
    random_array = (random_array*256).astype(np.uint)
    print('original array',random_array.shape, random_array)
    
    #compress and encode to base64
    #message = base64.b64encode(zlib.compress(random_array,9))
    message = array_2_base64(random_array)
    print('compressed message ',message)
        

    #base 64 decoding to byte stream, decompressing, and transforming to the same int type 
    #q = np.frombuffer(zlib.decompress(base64.decodebytes(message)), dtype=np.uint)
    decoded_array = base64_2_array(message, np.uint)
    print('decoded array', decoded_array.shape,decoded_array)
    
    pass
  
if __name__ == '__main__':
    test_base64_generation_from_array()