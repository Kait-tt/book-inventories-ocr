# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive

X_THRESHOLD = 40  # px
Y_THRESHOLD = 80  # px

interactive(True)


# 各列にいくつ黒要素があるかカウントする
# tw: 列の幅
def countBlack(ary, tw=2):
    h, w = ary.shape[:2]
    minh = int(h / 3 * 1)
    maxh = int(h / 3 * 2)
    cnts = []
    for sx in range(0, w - tw):
        cnt = 0
        for y in range(minh, maxh):
            if any([ary[y][x] == 255 for x in range(sx, sx + tw)]):
                cnt += 1
        cnts.append(cnt)
    return cnts
    

def myhough(threshold, minLineLength, maxLineGap):
    img = cv2.imread("./images/book_I_w.jpg")
    h, w = img.shape[:2]

    # gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # canny
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)

    # detect peek
    cnts = countBlack(edges, tw=2)
    plt.plot(range(0, len(cnts)), cnts)
    plt.savefig('./images/zf_hist0.png')    
    plt.show()
    
    divideBorder = 6
    divideLines = []
    divw = 20
    for x in range(w - divw - 1):
        if cnts[x] < divideBorder and all([cnts[tx] > divideBorder for tx in range(x + 1, x + divw)]):
            divideLines.append(x)

    # draw lines
    for x in divideLines:
        cv2.line(img, (x, 0), (x, h), (0, 0, 255), 2)
        if divideLines[0] < 1:
            dst = img[0:h,0:x]
        else:
            dst = img[0:h,x:divideLines[x+1]]
            strx = str(x)
            strh = str(h)
        cv2.imwrite('./images/'+"dst_" + strx +","+ strh +".jpg" ,dst)

    cv2.imshow("threshold:{0}, minLineLength:{1}, maxLineGap:{2}".format(threshold, minLineLength, maxLineGap), img)
    cv2.imwrite("./images/book_I_w_divided_6_20_0.jpg",img)
    

myhough(10, 10, 30)

cv2.waitKey(0)
cv2.destroyAllWindows()
