import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from .bayer import bayer_downsample
from .metrics import mse, psnr

# Returns a new image with 
#     where channels other than `channel` is filled with 0
#
def color_channel(img, channel):
    img = img.copy()
    img[:,:,[(channel+1)%3, (channel+2)%3]] = 0
    return img

# Displays 1 color image
def show_image(image, description='', bgr2rgb=False):
    img = plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB) if bgr2rgb else image, interpolation='none')
    plt.axis('off')
    plt.title(description)

# Displays >=1 color image 
def show_images(images, descriptions=[], layouts='', bgr2rgb=False):

    layouts = "{}1".format(len(images)) if layouts == '' else layouts
    descriptions = ['' for _ in range(len(images))] if descriptions == [] else descriptions

    for i,img in enumerate(images):
        plt.subplot("{}{}".format(layouts, i+1))
        # 1 channel -> 3 channel
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
        show_image(img, descriptions[i], bgr2rgb)

    plt.show()

def show_grayscale(img, description=''):
    if len(np.shape(img)) == 2 or np.shape(img)[2] == 1:
        img = np.tile(img[:,:,np.newaxis],(1,1,3))

    img = plt.imshow(img, interpolation='none')
    plt.axis('off')
    plt.title(description)
    

def show_grayscales(images, descriptions=[], layouts=''):

    layouts = "{}1".format(len(images)) if layouts == '' else layouts
    descriptions = ['' for _ in range(len(images))] if descriptions == [] else descriptions
    
    plt.figure(figsize=(20,20))
    for i,img in enumerate(images):
        plt.subplot("{}{}".format(layouts, i+1))
        show_grayscale(img, descriptions[i])
    plt.show()
    

# Compare two demosaicing functions on `img` 
def demosaic_compare(img, fs, crop=None):
    assert(len(fs) == 2)
    
    out = [img.copy()]
    mses = [np.Inf]
    psnrs = [np.Inf]
    
    downsampled = bayer_downsample(img)
    downsampled = np.sum(downsampled, axis=2, dtype=np.uint8)
    
    for i,f in enumerate(fs):
        demosaiced = f(downsampled)
        out.append(demosaiced)              
        mses.append(mse(out[0], demosaiced))
        psnrs.append(psnr(out[0], demosaiced))
    
    print(f"methods: {'1':>20} {'2':>20}")
    print(f"psnr:    {psnrs[1]:>20.3f} {psnrs[2]:>20.3f}")
    print(f"mses:    {mses[1]:>20.3f} {mses[2]:>20.3f}")

    out = list(map(lambda x: x/255., out))
    imgs = out + [np.squeeze(x) for x in [np.abs(out[1]-out[0]), np.abs([out[2]-out[0]])]]
    imgs = [(np.clip(img*255,0,255)).astype(np.uint8) for img in imgs]
    descriptions = ['original','1','2','|2-1|','|3-1|']

    if crop is not None:
        imgs = [img[(*crop,)] for img in imgs]

    # show_images(imgs[:3],  descriptions[:3],  '13', bgr2rgb=True)
    # show_images(imgs[-2:], descriptions[-2:], '12', bgr2rgb=True)
    show_images(imgs,  descriptions,  '15', bgr2rgb=True)