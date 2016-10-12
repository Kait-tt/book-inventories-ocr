# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 13:47:25 2016

@author: s1321168
"""

import cv2
import numpy as np

X_THRESHOLD = 20  # px
Y_THRESHOLD = 50 # px


def myhough(threshold, minLineLength, maxLineGap):
    img = cv2.imread("./images/book.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    cv2.imshow("canny", edges)

    if lines is None:
        print('lines not found')
        return

    print(len(lines))
    cnt = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < X_THRESHOLD / 180 * np.pi and abs(y1 - y2) > Y_THRESHOLD:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("threshold:{0}, minLineLength:{1}, maxLineGap:{2}".format(threshold, minLineLength, maxLineGap), img)
            cnt += 1
    print(cnt)

    cv2.imshow("threshold:{0}, minLineLength:{1}, maxLineGap:{2}".format(threshold, minLineLength, maxLineGap), img)

# for t in range(0, 100, 20):
#     for mlen in range(0, 200, 40):
#         for mgap in range(0, 30, 10):
#             myhough(idx, t, mlen, mgap)

myhough(20, 10, 10)

cv2.waitKey(0)
cv2.destroyAllWindows()
