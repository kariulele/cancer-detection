#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib.image import imread
import matplotlib.pyplot as plt



def show_image(imgpath, frame_color, gray=True):
    im = imgpath
    if gray:
        plt.imshow(im, cmap='gray')
    else:
        plt.imshow(im)
    h, w = im.shape[:2]
    plt.plot([0, 0, w, w, 0], [0, h, h, 0, 0], frame_color, linewidth = 2)
    plt.axis('off')

def show_test_res(clf, test_image,test_x,test_y):
    
    nelem = len(test_image)   # number of elements to show
    # reduce the margins
    plt.subplots_adjust(wspace = 0, hspace = 0,
                        top = 0.99, bottom = 0.01, left = 0.01, right = 0.99)
    
    plt.figure(figsize=(10,30))
    no = 1  # index current of subfigure
    for ii in range(nelem):
        plt.subplot((nelem // 3) + 1 , 3, ii+1)
        val_img_i = test_image[ii]
        x_val_i = test_x[ii]
        y_pred_i = clf.predict(x_val_i.reshape(1,-1))
        expected = test_y[ii]
        classname = "Cancer" if test_y[ii] else "No Cancer"
        show_image(val_img_i, 'g' if y_pred_i == expected else 'r')
        plt.title(classname + " " + ("OK" if y_pred_i == expected else "ERR"))
    
    plt.show()