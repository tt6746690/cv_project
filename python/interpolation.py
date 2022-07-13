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
#       https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1207407

# The algorithm
#   - interpolate `G`
#   - compute hue for `R`,`B` channels at subsampled locations
#   - interpolate hue for all pixels in `R`,`B` channels
#   - determine chrominance `R`,`B` from hue
# 
def demosaic_smooth_hue(img: np.ndarray, log_space=True) -> np.ndarray:
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
    if log_space:
        hue[(*B_idx,0)] = interp[(*B_idx,0)] - interp[(*B_idx,1)]
        hue[(*R_idx,1)] = interp[(*R_idx,2)] - interp[(*R_idx,1)]
    else:
        interp[(*B_idx,1)] += np.finfo(np.float32).eps
        interp[(*R_idx,1)] += np.finfo(np.float32).eps
        hue[(*B_idx,0)] = interp[(*B_idx,0)] / interp[(*B_idx,1)]
        hue[(*R_idx,1)] = interp[(*R_idx,2)] / interp[(*R_idx,1)]

    # interpolate hue
    hue[:,:,0] = cv.filter2D(hue[:,:,0], cv.CV_32F, K_B)
    hue[:,:,1] = cv.filter2D(hue[:,:,1], cv.CV_32F, K_R)

    # Compute chrominance B,R
    if log_space:
        interp[:,:,0] = np.clip(hue[:,:,0] + interp[:,:,1],0,1)
        interp[:,:,2] = np.clip(hue[:,:,1] + interp[:,:,1],0,1)
    else:
        interp[:,:,0] = np.clip(hue[:,:,0] * interp[:,:,1],0,1)
        interp[:,:,2] = np.clip(hue[:,:,1] * interp[:,:,1],0,1)

    interp = (interp*255).astype(np.uint8)

    return interp


# Smooth-hue Interpolation with median filtering (Freeman)
#       https://patents.google.com/patent/US4724395A/en
#       https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1207407
#
# The algorithm
#   - interpolate `G`,`R`,`B` channels
#   - compute hue, i.e. `(R-G,B-G)`
#   - median filter the hue
#   - determine chrominance `R`, `B` from hue
#
def demosaic_median_filter(img: np.ndarray, log_space=True) -> np.ndarray:
    assert(len(img.shape) in [2,3])

    if len(img.shape) == 2:
        img = bayer_split(img)

    K_B = 1/4 * np.array([ [1,2,1], [2,4,2], [1,2,1] ])
    K_G = 1/4 * np.array([ [0,1,0], [1,4,1], [0,1,0] ])
    K_R = K_B

    interp = img.copy().astype(np.float32) / 255.
    B_idx, G_idx, R_idx = bayer_idx(interp.shape[:2])

    # interpolate all channels
    interp[:,:,0] = cv.filter2D(interp[:,:,0], cv.CV_32F, K_B)
    interp[:,:,1] = cv.filter2D(interp[:,:,1], cv.CV_32F, K_G)
    interp[:,:,2] = cv.filter2D(interp[:,:,2], cv.CV_32F, K_R)

    # compute hue
    hue = np.zeros((*interp.shape[:2],2), dtype=np.float32)
    if log_space:
        hue[:,:,0] = interp[:,:,0] - interp[:,:,1]
        hue[:,:,1] = interp[:,:,2] - interp[:,:,1]
    else:
        interp[:,:,1] += np.finfo(np.float32).eps
        hue[:,:,0] = interp[:,:,0] / interp[:,:,1]
        hue[:,:,1] = interp[:,:,2] / interp[:,:,1]

    # median filter
    hue[:,:,0] = cv.medianBlur(hue[:,:,0], 5)
    hue[:,:,1] = cv.medianBlur(hue[:,:,1], 5)

    # adjustments
    if log_space:
        interp[:,:,0] = np.clip(hue[:,:,0] + interp[:,:,1],0,1)
        interp[:,:,2] = np.clip(hue[:,:,1] + interp[:,:,1],0,1)
    else:
        interp[:,:,0] = np.clip(hue[:,:,0] * interp[:,:,1],0,1)
        interp[:,:,2] = np.clip(hue[:,:,1] * interp[:,:,1],0,1)

    interp = (interp*255).astype(np.uint8)
    return interp


# Laplacian-corrected linear filter (MATLAB's demosaic)
#       https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1326587
#       https://www.ipol.im/pub/art/2011/g_mhcd/article.pdf
# 
def demosaic_laplacian_corrected(img, gain_factors=(1/2,5/8,3/4)):
    assert(len(img.shape) in [2,3])

    if len(img.shape) == 2:
        img = bayer_split(img)
        
    alpha,beta,gamma = gain_factors

    K_L = 1/4 * np.array([
        [0,0,-1,0,0],
        [0,0,0,0,0],
        [-1,0,4,0,-1],
        [0,0,0,0,0],
        [0,0,-1,0,0],
    ])

    K_B = 1/4 * np.array([ [1,2,1], [2,4,2], [1,2,1] ])
    K_G = 1/4 * np.array([ [0,1,0], [1,4,1], [0,1,0] ])
    K_R = K_B

    interp = img.copy().astype(np.float32) / 255.
    laplacian = interp.copy()
    B_idx, G_idx, R_idx = bayer_idx(interp.shape[:2])

    # Interpolate R,G,B
    interp[:,:,0] = cv.filter2D(interp[:,:,0], cv.CV_32F, K_B)
    interp[:,:,1] = cv.filter2D(interp[:,:,1], cv.CV_32F, K_G)
    interp[:,:,2] = cv.filter2D(interp[:,:,2], cv.CV_32F, K_R)

    # Compute discrete laplacian in 5x5 neighborhood
    laplacian[:,:,0] = cv.filter2D(laplacian[:,:,0], cv.CV_32F, K_L)
    laplacian[:,:,1] = cv.filter2D(laplacian[:,:,1], cv.CV_32F, K_L)
    laplacian[:,:,2] = cv.filter2D(laplacian[:,:,2], cv.CV_32F, K_L)

    # Laplacian correction
    # Green
    interp[(*R_idx,1)] = interp[(*R_idx,1)] + alpha*laplacian[(*R_idx,2)]
    interp[(*B_idx,1)] = interp[(*B_idx,1)] + alpha*laplacian[(*B_idx,0)]
    # Red
    interp[(*G_idx,2)] = interp[(*G_idx,2)] + beta*laplacian[(*G_idx,1)]
    interp[(*B_idx,2)] = interp[(*B_idx,2)] + beta*laplacian[(*B_idx,0)]
    # Blue
    interp[(*G_idx,0)] = interp[(*G_idx,0)] + gamma*laplacian[(*G_idx,1)]
    interp[(*R_idx,0)] = interp[(*R_idx,0)] + gamma*laplacian[(*R_idx,2)]

    interp = (np.clip(interp*255,0,255)).astype(np.uint8)

    return interp
