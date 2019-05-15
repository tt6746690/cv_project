from typing import Tuple

import cv2 as cv
import numpy as np

# Given full-res image `I`
#    Return indices to R,B,G channels according to the following pattern
#
#    R  G
#    G  B
# 
def bayer_idx(shape: Tuple[int,int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert(len(shape) == 2)
    assert(all(np.array(shape) % 2 == np.zeros(2)))

    rsize = (np.array(shape) / 2).astype(np.int)
    
    rowix = np.arange(1,shape[0],2)[:,np.newaxis]
    colix = np.arange(1,shape[1],2)[np.newaxis,:]
    B_idx = np.tile(rowix, (1,rsize[1])), np.tile(colix, (rsize[0],1))

    rowix = np.hstack((np.arange(1,shape[0],2)[:,np.newaxis], \
                       np.arange(0,shape[0],2)[:,np.newaxis]))
    colix = np.arange(0,shape[1])
    G_idx = np.tile(rowix, (1,rsize[1])), np.tile(colix, (rsize[0],1))

    rowix = np.arange(0,shape[0],2)[:,np.newaxis]
    colix = np.arange(0,shape[1],2)[np.newaxis,:]
    R_idx = np.tile(rowix, (1,rsize[1])), np.tile(colix, (rsize[0],1))
    
    return B_idx, G_idx, R_idx
    
# Get masks for bayer mosaics,
#     where `shape` is a 2-tuple
def bayer_mask(shape: Tuple[int,int]) -> np.ndarray:
    B_idx, G_idx, R_idx = bayer_idx(shape)
    mask = np.zeros((*shape, 3), dtype=np.uint8)

    mask[(*B_idx,0)] = 1
    mask[(*G_idx,1)] = 1
    mask[(*R_idx,2)] = 1

    return mask
    
# Downsampling `img` according to `RGGB` pattern
# 
def bayer_downsample(img: np.ndarray) -> np.ndarray:
    mask = bayer_mask(img.shape[:2])
    c = img * mask
    return c


# Split image into 3 channels according to bayer pattern
#       mxnx1 -> mxnx3
def bayer_split(img: np.ndarray) -> np.ndarray:
    assert(len(img.shape) == 2)
    
    shape = img.shape
    B_idx, G_idx, R_idx = bayer_idx(shape)

    out = np.zeros((*shape, 3), dtype=np.uint8)
    out[(*B_idx,0)] = img[B_idx]
    out[(*G_idx,1)] = img[G_idx]
    out[(*R_idx,2)] = img[R_idx]

    return out