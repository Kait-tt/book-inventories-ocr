# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 13:42:56 2016

@author: s1321168
"""

import cv2
import numpy as np

img = cv2.imread("C:/image/BIloop[91]_.JPG")

for i in range(100):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150 + 10 * i,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
    
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.imwrite("C:/image/BIloop[91]_loop"+str([i])+".jpg",img)        


