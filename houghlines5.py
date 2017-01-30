# -*- coding: utf-8 -*-

import cv2
import numpy as np

img = cv2.imread("./images/book.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,200,200,apertureSize = 3)
minLineLength = 300
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=minLineLength, maxLineGap=maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    
    
    cv2.imwrite("./images/bookhough5.jpg",img)
    cv2.imshow('img' , img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def removeNotPeekArea(ary, peeks, pw=2):
    h, w = ary.shape[:2]
    ary2 = np.zeros_like(ary)
    for x in peeks:
        for tx in range(-pw, pw + 1):
            if 0 <= x + tx < w:
                for y in range(h):
                    ary2[y][x + tx] = ary[y][x + tx]
    return ary2