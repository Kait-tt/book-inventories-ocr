# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 13:47:25 2016

@author: s1321168
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive

X_THRESHOLD = 20  # px
Y_THRESHOLD = 50 # px

interactive(True)


def myhough(threshold, minLineLength, maxLineGap):
    img = cv2.imread("./images/book.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)

    h = len(edges)
    w = len(edges[0])
    tw = 2  # px
    minh = int(h / 3 * 1)
    maxh = int(h / 3 * 2)
    threshold2 = 70
    cnts = []
    for sx in range(0, w - tw):
        cnt = 0
        for y in range(minh, maxh):
            if any([edges[y][x] == 255 for x in range(sx, sx + tw)]):
                cnt += 1
        cnts.append(cnt)

    # detection peak
    peeks = []
    pw = 20
    peek_v = 200
    for i in range(len(cnts) - pw):
        if all([cnts[i - j] <= cnts[i] for j in range(-pw, pw + 1) if j != 0]):
            peeks.append(peek_v)
        else:
            peeks.append(0)


    plt.plot(range(0, len(cnts)), cnts)
    plt.plot(range(0, len(peeks)), peeks)
    plt.show()

    cv2.imshow("canny", edges)

    for sx in range(0, w - tw):
        if cnts[sx] < threshold2:
            for y in range(0, h):
                for x in range(sx, sx + tw):
                    edges[y][x] = 0

    cv2.imshow("canny2", edges)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

    if lines is None:
        print('lines not found')
        return

    # hough lines
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     if abs(x1 - x2) < X_THRESHOLD / 180 * np.pi and abs(y1 - y2) > Y_THRESHOLD:
    #         cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # peek lines
    for i in range(len(peeks)):
        if peeks[i] == peek_v:
            cv2.line(img, (i, 0), (i, h), (0, 0, 255), 2)

    cv2.imshow("threshold:{0}, minLineLength:{1}, maxLineGap:{2}".format(threshold, minLineLength, maxLineGap), img)

# for t in range(0, 100, 20):
#     for mlen in range(0, 200, 40):
#         for mgap in range(0, 30, 10):
#             myhough(idx, t, mlen, mgap)

myhough(10, 10, 30)

cv2.waitKey(0)
cv2.destroyAllWindows()
