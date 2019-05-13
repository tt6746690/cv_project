import cv2 as cv
import numpy as np

# Given full-res image `I`
#    Return indices to R,B,G channels according to the following pattern
#
#    R  G
#    G  B
# 
def bayer_idx(shape):
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
def bayer_mask(shape):
    B_idx, G_idx, R_idx = bayer_idx(shape)

    B_mask = np.zeros(shape,dtype=np.uint8)
    G_mask = np.zeros(shape,dtype=np.uint8)
    R_mask = np.zeros(shape,dtype=np.uint8)

    B_mask[B_idx] = 1
    G_mask[G_idx] = 1
    R_mask[R_idx] = 1

    return B_mask, G_mask, R_mask
    
    
# Downsampling `img` according to `RGGB` pattern
# 
def bayer_downsample(img):
    B_mask,G_mask,R_mask = bayer_mask(img.shape[:2])

    c = img.copy()
    c[:,:,0] = c[:,:,0] * B_mask
    c[:,:,1] = c[:,:,1] * G_mask
    c[:,:,2] = c[:,:,2] * R_mask
    
    return c