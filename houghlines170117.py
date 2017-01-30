# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive

X_THRESHOLD = 40  # px
Y_THRESHOLD = 80  # px

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


def removeNotPeekArea(ary, peeks, pw=2):
    h, w = ary.shape[:2]
    ary2 = np.zeros_like(ary)
    for x in peeks:
        for tx in range(-pw, pw + 1):
            if 0 <= x + tx < w:
                for y in range(h):
                    ary2[y][x + tx] = ary[y][x + tx]
    return ary2


def myhough(threshold, minLineLength, maxLineGap):
    img = cv2.imread("./images/BI12_iso800triro.jpg")
    h, w = img.shape[:2]

    # gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # scale out
    scale = 0.5
    gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

    # canny
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)
    # cv2.imshow("canny2", edges)
    cv2.imshow("canny2", cv2.resize(edges, (w, h)))

    # detect peek
    cnts = countBlack(edges, tw=2)
    peeks = detectPeek(cnts, pw=int(20*scale))
    plt.plot(range(0, len(cnts)), cnts)
    plt.plot(peeks, [cnts[x] for x in peeks], 'o')
    plt.show()

    # remove not peek area
    edges = removeNotPeekArea(edges, peeks, pw=2)
    # cv2.imshow("remove not peek area", edges)
    cv2.imshow("remove not peek area", cv2.resize(edges, (w, h)))

    # hough
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    if lines is None:
        print('lines not found')
        return

    # draw hough lines only vertically line
    for line in lines:
        x1, y1, x2, y2 = [int(a / scale) for a in line[0]]
        if abs(x1 - x2) < X_THRESHOLD / 180 * np.pi and abs(y1 - y2) > Y_THRESHOLD:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # draw peek lines
    for x in [int(x / scale) for x in peeks]:
        cv2.line(img, (x, 0), (x, h), (0, 0, 255), 2)

    cv2.imshow("threshold:{0}, minLineLength:{1}, maxLineGap:{2}".format(threshold, minLineLength, maxLineGap), img)

myhough(10, 10, 30)

cv2.waitKey(0)
cv2.destroyAllWindows()