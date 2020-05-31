import cv2 as cv
import numpy as np

# Add Gaussian additive white noise to image
#     assumes `img` has `dtype=np.uint8`
#
#     https://gist.github.com/Prasad9/28f6a2df8e8d463c6ddd040f4f6a028a
#
def additive_gaussian_noise(img, mean=0, variance=1, centered=False):
    img = img.copy() / 255.
    noise = np.random.normal(mean, variance**0.5, img.shape).reshape(img.shape) / 255.
    if centered:
        img2 = 2*img
        mul_noise_down = img2*(1 + noise)
        mul_noise_up   = (1-img2+1)*(1 + noise)*-1 + 2
        added = np.clip(np.where(img2 <= 1, mul_noise_down, mul_noise_up)/2.,0,1)
    else:
        added = np.clip(img+noise,0,1)
    added = (added*255).astype(np.uint8)
    return added
