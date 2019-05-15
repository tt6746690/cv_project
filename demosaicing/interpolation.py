import cv2 as cv
import numpy as np

from .bayer import bayer_split, bayer_idx

# Bilinear Interpolation
# 
def demosaic_bilinear(img: np.ndarray) -> np.ndarray:
    assert(len(img.shape) in [2,3])

    if len(img.shape) == 2:
        img = bayer_split(img)
    
    K_B = 1/4 * np.array([ [1,2,1], [2,4,2], [1,2,1] ])
    K_G = 1/4 * np.array([ [0,1,0], [1,4,1], [0,1,0] ])
    K_R = K_B

    interp = img.copy()
    interp[:,:,0] = cv.filter2D(img[:,:,0], cv.CV_8U, K_B)
    interp[:,:,1] = cv.filter2D(img[:,:,1], cv.CV_8U, K_G)
    interp[:,:,2] = cv.filter2D(img[:,:,2], cv.CV_8U, K_R)

    return interp


# Smooth-hue Interpolation
#       https://patents.google.com/patent/US4642678A/en
# 
# The algorithm
#   - interpolate `G`
#   - compute hue for `R`,`B` channels at subsampled locations
#   - interpolate hue for all pixels in `R`,`B` channels
#   - determine chrominance `R`,`B` from hue
# 
def demosaic_smooth_hue(img: np.ndarray) -> np.ndarray:
    assert(len(img.shape) in [2,3])

    if len(img.shape) == 2:
        img = bayer_split(img)

    K_B = 1/4 * np.array([ [1,2,1], [2,4,2], [1,2,1] ])
    K_G = 1/4 * np.array([ [0,1,0], [1,4,1], [0,1,0] ])
    K_R = K_B

    interp = img.copy().astype(np.float32) / 255.
    B_idx, G_idx, R_idx = bayer_idx(interp.shape[:2])

    # interpolate luminance G
    interp[:,:,1] = cv.filter2D(interp[:,:,1], cv.CV_32F, K_G)

    # compute hue B: [:,:,0] R: [:,:,1]
    hue = np.zeros((*interp.shape[:2],2), dtype=np.float32)
    hue[(*B_idx,0)] = interp[(*B_idx,0)] / interp[(*B_idx,1)]
    hue[(*R_idx,1)] = interp[(*R_idx,2)] / interp[(*R_idx,1)]

    # interpolate hue
    hue[:,:,0] = cv.filter2D(hue[:,:,0], cv.CV_32F, K_B)
    hue[:,:,1] = cv.filter2D(hue[:,:,1], cv.CV_32F, K_R)

    # Compute chrominance B,R
    interp[:,:,0] = np.clip(hue[:,:,0] * interp[:,:,1],0,1)
    interp[:,:,2] = np.clip(hue[:,:,1] * interp[:,:,1],0,1)

    interp = (interp*255).astype(np.uint8)

    return interp