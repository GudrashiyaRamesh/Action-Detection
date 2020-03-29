# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:45:37 2020

@author: RAMESH
"""

import pandas as pd
import numpy as np
import cv2


video = cv2.VideoCapture("130.mp4")
i = 0
c = 7
count = 0
while(video.isOpened()):
    
    ret , frame = video.read()
    if ret == True:
        if c == 0:
            str = f"/frames/img{i}.jpg"
            cv2.imwrite(str,frame)
            i += 1
            count +=1
            c = 7
            
            if count == 31:
                break
        c -= 1
    else:
        break
            
video.release()
cv2.destroyAllWindows()

video = cv2.VideoCapture("130.mp4")
c = 7
images = []
count = 0
while(video.isOpened()):
    
    ret , frame = video.read()
    if ret == True:
        if c == 0:
           # frame = preprocess_image(frame)
            images.append(frame)
            count +=1
            c = 7
            if count == 30:
                break
        c -= 1
    else:
        break
            
video.release()
cv2.destroyAllWindows()

print(len(images))
