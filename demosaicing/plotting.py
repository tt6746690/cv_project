import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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
