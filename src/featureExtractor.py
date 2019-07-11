#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

i1 = storage.allList[10].content

def image_to_feature(img):
    gray= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    detector = cv2.AKAZE_create()
    (kps1, descs1) = detector.detectAndCompute(gray, None)
    return descs1



plt.imshow(image_to_feature(i1))