import os
import numpy as np
import cv2 as cv

from .metrics import psnr
from .bayer import bayer_downsample

def kodak_dataset(kodak_dir="../data/kodak/"):
    kodak_images = list(map(lambda x: f'{kodak_dir}{x}', os.listdir(kodak_dir)))
    kodak_images = sorted(kodak_images)
    kodak_images = filter(lambda x: x.endswith('.png'), kodak_images)
    kodak = list(map(lambda x: cv.imread(x, -1), kodak_images))
    return kodak

# Run `demosaicing_algorithm` algorithm over entire kodak dataset
def validate_kodak(demosaicing_algorithm):

    kodak = kodak_dataset()
    history = {
        'demosaiced': [],
        'psnr': []
    }

    for i, img in enumerate(kodak):
        downsampled = bayer_downsample(img)
        downsampled = np.sum(downsampled, axis=2, dtype=np.uint8)

        demosaiced = demosaicing_algorithm(downsampled)
        psnrv = psnr(img, demosaiced)

        history['demosaiced'].append(demosaiced)
        history['psnr'].append(psnrv)
        
    return history