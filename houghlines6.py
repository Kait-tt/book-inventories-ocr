# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive

X_THRESHOLD = 20  # px
Y_THRESHOLD = 50  # px

interactive(True)


def countBlack(ary, tw=2):
    h, w = ary.shape[:2]
    minh = int(h / 3 * 1)
    maxh = int(w / 3 * 2)
    cnts = []
    for sx in range(0, w - tw):
        cnt = 0
        for y in range(minh, maxh):
            if any([ary[y][x] == 255 for x in range(sx, sx + tw)]):
                cnt += 1
        cnts.append(cnt)
    return cnts


def detectPeek(ary, pw=20):
    peeks = []
    for i in range(len(ary) - pw):
        if all([ary[i - j] <= ary[i] for j in range(-pw, pw + 1) if j != 0]):
            peeks.append(i)
    return peeks


def myhough(threshold, minLineLength, maxLineGap):
    img = cv2.imread("./images/book.jpg")

    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)
    cv2.imshow("canny", edges)

    cnts = countBlack(edges, tw=2)
    plt.plot(range(0, len(cnts)), cnts)

    peeks = detectPeek(cnts, pw=20)
    plt.plot(peeks, [cnts[x] for x in peeks], 'o')
    plt.show()

    # hough
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

    if lines is None:
        print('lines not found')
        return

    # draw hough lines only vertically line
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < X_THRESHOLD / 180 * np.pi and abs(y1 - y2) > Y_THRESHOLD:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # draw peek lines
    for w in peeks:
        cv2.line(img, (w, 0), (w, h), (0, 0, 255), 2)

    cv2.imshow("threshold:{0}, minLineLength:{1}, maxLineGap:{2}".format(threshold, minLineLength, maxLineGap), img)

myhough(10, 10, 30)

cv2.waitKey(0)
cv2.destroyAllWindows()
