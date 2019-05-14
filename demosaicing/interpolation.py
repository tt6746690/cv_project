import cv2 as cv
import numpy as np

from .bayer import bayer_idx


# Bilinear Interpolation
# 
def demosaic_bilinear(img: np.ndarray) -> np.ndarray:
    assert(len(img.shape) in [2,3])

    if len(img.shape) == 2:

        shape = img.shape
        img1 = img
        B_idx, G_idx, R_idx = bayer_idx(shape)

        img = np.zeros((*shape, 3), dtype=np.uint8)
        img[(*B_idx,0)] = img1[(*B_idx,0)] 
        img[(*G_idx,1)] = img1[(*G_idx,1)] 
        img[(*R_idx,2)] = img1[(*R_idx,2)] 
    
    K_B = 1/4 * np.array([ [1,2,1], [2,4,2], [1,2,1] ])
    K_G = 1/4 * np.array([ [0,1,0], [1,4,1], [0,1,0] ])
    K_R = K_B

    interp = img.copy()
    interp[:,:,0] = cv.filter2D(img[:,:,0], cv.CV_8U, K_B)
    interp[:,:,1] = cv.filter2D(img[:,:,1], cv.CV_8U, K_G)
    interp[:,:,2] = cv.filter2D(img[:,:,2], cv.CV_8U, K_R)

    return interp
