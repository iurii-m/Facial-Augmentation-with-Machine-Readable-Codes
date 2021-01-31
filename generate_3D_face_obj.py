# -*- coding: utf-8 -*-
"""
Function to generate 3D face model from the input 2D face image

@author: iurii
"""


import sys
import argparse
import cv2
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils_3DDFA.serialization import ser_to_obj

from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from TDDFA_ONNX import TDDFA_ONNX

import os

def generate_3D_face_obj(img , name_id , modelsavepath, config='configs/mb1_120x120.yml'):
    
    # Given a still image path and load to BGR channel
    #img = cv2.imread(args.img_fp)
    
    config='configs/mb1_120x120.yml'
    cfg = yaml.load(open(config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, (default onnx flag)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'

    face_boxes = FaceBoxes_ONNX()
    tddfa = TDDFA_ONNX(**cfg)


    # Detect faces, get 3DMM params and roi boxes
    boxes = face_boxes(img)
    n = len(boxes)
    if n == 0:
        print(f'No face detected, exit')
        sys.exit(-1)
    print(f'Detect {n} faces')

    param_lst, roi_box_lst = tddfa(img, boxes)

    # Visualization and serialization
    dense_flag = 'obj'
    wfp = modelsavepath +'/'+ name_id + '.obj'
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
    ser_to_obj(img, ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
    
    
    
    

    
