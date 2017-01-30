# -*- coding: utf-8 -*-
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
from matplotlib.font_manager import FontProperties
import requests
import base64
import json
import Levenshtein
import pprint
import tkinter
import tkinter.filedialog
import pickle
import time

interactive(True)

#######################################
target_title = "プログラミング入門"
#######################################

entry_image = tkinter.filedialog.askopenfilename(title="Choose image file")


# 高さ/3から2/3までの各列にいくつ黒要素があるかカウントする
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


def myHough1():
    # start1 = time.time()
    img = Image.open(entry_image)
    # 90°回転
    img90PIL = img.transpose(Image.ROTATE_90)
    img90 = cv2.cvtColor(np.array(img90PIL), cv2.COLOR_RGB2BGR)
    lineimg1 = img90

    # 高さと幅の取得
    h, w = img90.shape[:2]

    # gray scale
    gray = cv2.cvtColor(img90, cv2.COLOR_BGR2GRAY)

    # canny
    Lowthre = 100
    Highthre = 150
    edges = cv2.Canny(gray, Lowthre, Highthre, apertureSize=3)
    cv2.imwrite('./images/cannyimages/' + str(Lowthre) + ',' + str(Highthre) + ',' + ',' + ".jpg", edges)

    # detect peek
    cnts = countBlack(edges, tw=2)
    pprint.pprint(cnts)
    # pprint.pprint(cnts)
    divideBorder = 8
    divideLines = []
    divideImages = []
    divw = 10

    for x in range(w - divw - 1):
        if cnts[x] < divideBorder and all([cnts[tx] > divideBorder for tx in range(x + 1, x + divw)]):
            divideLines.append(x)
    if len(divideLines) is not None:
        # if divideLines is not None:
        divideLines.insert(0, 0)
        divideLines.append(w)

    # dvidelines
    for x in range(len(divideLines) - 1):
        divide_img = img90[0:h, divideLines[x]:divideLines[x + 1]]
        divideImages.append(divide_img)
        # cv2.line(lineimg1, (divideLines[x], 0), (divideLines[x], h), (0, 0, 255), 16)
        cv2.imwrite('./images/lineimages/' + str(Lowthre) + ',' + str(Highthre) + ',' + ".jpg", lineimg1)
        cv2.imwrite('./images/myhough1/' + str(x) + '.jpg', divide_img)
    # print(np1)
    # elapsed_time1 = time.time() - start1
    # print("elapsed_time1:{0}".format(elapsed_time1))

    return divideImages, divideLines

myHough1()