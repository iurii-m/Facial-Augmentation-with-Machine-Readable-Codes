# -*- coding: utf-8 -*-
"""
usefull cv2 utils

@author: iurii
"""

import cv2
import math

def crop_face(face_image, faces_rect, face_center, modify_rect=True, d_mult = 1.0):
        """

        :param face_image:
        :param faces_rect: Tuple with the following values [x, y, width, height]
        :param face_center: Tuple with the following values [center_x, center_y]
        :param modify_rect:
        :return:
        """
        # Check if the face is empty
        if face_image is None or faces_rect is None:
            raise Exception("Empty Source")

        # Crop face using the rectangle region along the face center
        # If modify_rect is True, the function returns an image area cropped by a square with size of max(width, height)
        # multiplied by the scaling parameter
        if modify_rect:
            half_side_value = int(d_mult * max(faces_rect[2], faces_rect[3]) / 2)
            x = face_center[0] - half_side_value
            y = face_center[1] - half_side_value
            width = 2 * half_side_value
            height = 2 * half_side_value
        # If the modify_rect is False, simply crop the image using the input rectangle
        else:
            x = face_center[0] - (faces_rect[2] / 2)
            y = face_center[1] - (faces_rect[3] / 2)
            width = faces_rect[2]
            height = faces_rect[3]

        # Cast every thing to int because all the dimensions need to be integers
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)

        # Crop the image
        cropped_face = crop_any_rect(face_image, (x, y, width, height))

        return cropped_face

def crop_any_rect(src, roi):
        """
        Simple function that crops and image using the desired rectangle. If the rectangle lays outside of the image
        dimensions, the cropped image is filled with pixels similar to the border of the original image.
        :param src: An image loaded using opencv
        :param roi: Desired rectangle around the face. Needs to be in the following order: [x, y, width, height]
        :return: Returns the image cropped with the desired rectangle
        """
        bt_mrg = 0
        tp_mrg = 0
        lft_mrg = 0
        rgt_mrg = 0

        if roi[0] < 0:
            lft_mrg = abs(roi[0])

        if roi[1] < 0:
            tp_mrg = abs(roi[1])

        # Attention to this, ndarray.shape return (height, width) so the indexes in the src need to be switched
        gp_cols = src.shape[1] - roi[0] - roi[2]
        gp_rows = src.shape[0] - roi[1] - roi[3]

        if gp_cols < 0:
            rgt_mrg = abs(gp_cols)

        if gp_rows < 0:
            bt_mrg = abs(gp_rows)

        src = cv2.copyMakeBorder(src, tp_mrg, bt_mrg, lft_mrg, rgt_mrg, cv2.BORDER_CONSTANT, value=0)

        new_x = roi[0] + lft_mrg
        new_y = roi[1] + tp_mrg

        crop_img = src[new_y:new_y + roi[3], new_x:new_x + roi[2]]

        return crop_img
    

def distance_2_pts(p0, p1):
    """
    Function to measure distance between 2 points
    
    Parameters
    ----------
    p0 : tuple (x,y)
         first point
         
    p1 : tuple (x,y)
         second point

    Returns
    -------
    float
    distance between 2 points

    """
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = int(max(0, y)), int(min(img.shape[0], y + img_overlay.shape[0]))
    x1, x2 = int(max(0, x)), int(min(img.shape[1], x + img_overlay.shape[1]))

    # Overlay ranges
    y1o, y2o = int(max(0, -y)), int(min(img_overlay.shape[0], img.shape[0] - y))
    x1o, x2o = int(max(0, -x)), int(min(img_overlay.shape[1], img.shape[1] - x))

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return img
    
    # Exit if not equal ranges
    if y2o-y1o!=y2-y1 or x2o-x1o!=x2-x1:
        return img


    channels = img.shape[2]
  
    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha
    

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +alpha_inv * img[y1:y2, x1:x2, c])
        
    return img
