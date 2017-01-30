# -*- coding: utf-8 -*-
from PIL import Image
import os
import cv2
import glob
import numpy as np
from matplotlib import interactive
import matplotlib.pyplot as plt
import tkinter
import tkinter.filedialog
import requests
import pycurl
import base64
import json
import pprint

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

def myHough1():
    entry_image = tkinter.filedialog.askopenfilename(title="Choose image file")
    img = Image.open(entry_image)
    # 90°回転
    img90PIL = img.transpose(Image.ROTATE_90)
    img90 = cv2.cvtColor(np.array(img90PIL), cv2.COLOR_RGB2BGR)
        
    # 高さと幅の取得
    h, w = img90.shape[:2]

    # gray scale
    gray = cv2.cvtColor(img90, cv2.COLOR_BGR2GRAY)

    # canny
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)
    cv2.imwrite("./images/edges.jpg", edges)
    # detect peek
    cnts = countBlack(edges, tw=2)
    divideBorder = 15
    divideLines = []
    divideImages = []
    divw = 25
    myHough1imgSize = []
    np1 = []
    filecnt = 0

    plt.plot(range(0, len(cnts)), cnts)
    plt.savefig('./images/plot.png')
    plt.show()

    divideLines.append(0)
    for x in range(w - divw - 1):
        if cnts[x] < divideBorder and all([cnts[tx] > divideBorder for tx in range(x + 1, x + divw)]):
            divideLines.append(x)
    divideLines.append(w)
    # print(divideLines)
    # print(len(divideLines))
    # dvide lines
    for x in range(len(divideLines) - 1):
        # if x == -1:
        #     divide_img = img90[0:h, 0:divideLines[x+1]]
        #     myHough1imgSize = ([0, divideLines[x+1]])
        #     np1.append(myHough1imgSize)
        #     strx = str(0)
        # elif x == len(divideLines) - 1:
        #     divide_img = img90[0:h, divideLines[x]:w]
        #     myHough1imgSize = ([divideLines[x], w])
        #     cv2.line(img90, (divideLines[x], 0), (divideLines[x], h), (0, 0, 255), 2)
        #     np1.append(myHough1imgSize)
        #     strx = str(divideLines[x])
        # else:
        divide_img = img90[0:h, divideLines[x]:divideLines[x+1]]
        # myHough1imgSize = ([divideLines[x], divideLines[x + 1]])
        # cv2.line(img90, (divideLines[x], 0), (divideLines[x], h), (0, 0, 255), 2)
        # np1.append(myHough1imgSize)
        # strx = str(divideLines[x])
        # print(x)
        # cv2.imshow("divide_img", divide_img)
        divideImages.append(divide_img)
        # cv2.imwrite('./images/myHough1/' + str(filecnt) + ',' + strx + ".jpg", divide_img)
        # filecnt += 1
    #cv2.imwrite('./images/writelineimg.jpg', img90)
    #print(np1)
    return divideImages, divideLines
        
def myHough2():
    stageImages, stagePositions = myHough1()
    # MY_HOUGH1 = "./images/myHough1"
    # for infile in glob.glob(os.path.join(MY_HOUGH1, "*.jpg")):
    fragmentImages = []
    fragmentPositions = []
    for i, img in enumerate(stageImages):
        # path, ext = os.path.splitext(infile)
        # path, ext = os.path.splitext(os.path.basename(infile))
        # img2 = Image.open(img)

        # img2 = Image.fromarray(img)
        img2 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # im_list = np.asarray(img)

        # plt.imshow(im_list)
        # plt.show()
        # img2 = Image.fromarray(img)
        # img2.save('./images/' + str(i) + '.jpg')

        # img2.show()

        # 270°回転
        img270PIL = img2.transpose(Image.ROTATE_270)
        img270 = cv2.cvtColor(np.array(img270PIL), cv2.COLOR_RGB2BGR)

        # gray scale
        gray = cv2.cvtColor(img270, cv2.COLOR_BGR2GRAY)

        # canny
        edges = cv2.Canny(gray, 100, 150, apertureSize=3)

        # detect peek
        h, w = img270.shape[:2]
        cnts = countBlack(edges, tw=2)
        divideBorder = 8
        divideLines2 = []
        divw = 10
        DvideLinesImg = []
        # myHough2imgSize = []
        np2 = []

        # if len(divideLines2) == 0:
        #     # 後で何か書く
        #     continue
        divideLines2.append(0)
        for x in range(w - divw - 1):
            if cnts[x] < divideBorder and all([cnts[tx] > divideBorder for tx in range(x + 1, x + divw)]):
                divideLines2.append(x)
        divideLines2.append(w)
        # print(divideLines2)
        # print(len(divideLines2))
        #dvideLines

        print(i,'段目')
        print(divideLines2)
        print(stagePositions[i])
        for y in range(0, len(divideLines2) - 1):
            # if y == len(divideLines2) - 1:
            #     divide_img = img270[0:h, divideLines2[y]:w]
            #     stry = str(y)
            #     myHough2imgSize = np.array([y][divideLines2[y]][w])
            #     np2.append(myHough2imgSize)
            #     strd = str(divideLines2[y])
            #     cv2.imwrite('./images/myHough2/' + path + ',' + strd + ".jpg", divide_img)
            # else:
            divide_img = img270[0:h, divideLines2[y]:divideLines2[y + 1]]
            # stry = str(y)
            # myHough2imgSize = np.array([y][divideLines2[y]][divideLines2[y + 1]])
            # strd= str(divideLines2[y])
            # np2.append(myHough2imgSize)
            # cv2.imwrite('./images/myHough2/' + path + ',' + strd + ".jpg", divide_img)
            fragmentImages.append(divide_img)
            fragmentPos = {}
            fragmentPos['x1'] = divideLines2[y]
            fragmentPos['y1'] = stagePositions[i]
            fragmentPos['x2'] = divideLines2[y + 1]
            fragmentPos['y2'] = stagePositions[i + 1]
            fragmentPositions.append(fragmentPos)
    print(fragmentPositions)
        # print(fragmentImages)
    return fragmentImages, fragmentPositions
#myHough1()
myHough2()

GOOGLE_CLOUD_VISION_API_URL = 'https://vision.googleapis.com/v1/images:annotate?key='
API_KEY = "AIzaSyDPv1cYnwwjxAlw4uy0aPhu9634sh7LxRM"
def google_cloud_vison():
    GCV_OBJ = []
    fragmentImages, fragmentPositions = myHough2()
    # MY_HOUGH2 = "./images/myHough2"
    # for file in glob.glob(os.path.join(MY_HOUGH2, "*.jpg")):
    #     stream = open(file, 'rb').read()
    for i, img in enumerate(fragmentImages):
        # base64でencode
        infile = base64.b64encode(img)
        # base64.b64encode(infile).decode("utf-8")
        # image_path, ext = os.path.splitext(os.path.basename(infile))

        api_url = GOOGLE_CLOUD_VISION_API_URL + API_KEY
        req_body = json.dumps({
            'requests': [{
                'image': {
                    'content': str(base64.b64encode(infile).decode("utf-8"))
                },
                'features': [{
                    'type': 'TEXT_DETECTION',
                    'maxResults': 1
                }],
                'imageContext': {
                    'languageHints': ['ja']
                }
            }]
        })
    res = requests.post(api_url, data=req_body)
    GCV_OBJ.append(res.json)
    pprint.pprint(GCV_OBJ)
    return GCV_OBJ


google_cloud_vison()
# OCR_OBJ = google_cloud_vison()
# print(OCR_OBJ)



