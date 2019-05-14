import cv2 as cv
import numpy as np


# Bilinear Interpolation
# 
def demosaic_bilinear(img):
    assert(len(img.shape) == 3)
    
    K_B = 1/4 * np.array([ [1,2,1], [2,4,2], [1,2,1] ])
    K_G = 1/4 * np.array([ [0,1,0], [1,4,1], [0,1,0] ])
    K_R = K_B

    interp = img.copy()
    interp[:,:,0] = cv.filter2D(img[:,:,0], cv.CV_8U, K_B)
    interp[:,:,1] = cv.filter2D(img[:,:,1], cv.CV_8U, K_G)
    interp[:,:,2] = cv.filter2D(img[:,:,2], cv.CV_8U, K_R)

    return interp