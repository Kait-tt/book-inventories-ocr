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
        divide_img = img90[0:h, divideLines[x]:divideLines[x+1]]
        divideImages.append(divide_img)
        cv2.line(lineimg1, (divideLines[x], 0), (divideLines[x], h), (0, 0, 255), 8)
        cv2.imwrite('./images/lineimages/' + str(Lowthre) + ',' + str(Highthre) + ',' + ".jpg", lineimg1)
        cv2.imwrite('./images/myhough1/' + str(x) + '.jpg', divide_img)
    #print(np1)
    # elapsed_time1 = time.time() - start1
    # print("elapsed_time1:{0}".format(elapsed_time1))

    return divideImages, divideLines
        
def myHough2():
    # start2 = time.time()
    stageImages, stagePositions = myHough1()

    # MY_HOUGH1 = "./images/myHough1"
    # for infile in glob.glob(os.path.join(MY_HOUGH1, "*.jpg")):

    stageAndfragmentPositions = []
    stageAndfragmentImages = []

    for i, img in enumerate(stageImages):
        img2 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 270°回転
        img270PIL = img2.transpose(Image.ROTATE_270)
        img270 = cv2.cvtColor(np.array(img270PIL), cv2.COLOR_RGB2BGR)
        lineimg2 = img270

        # gray scale
        gray = cv2.cvtColor(img270, cv2.COLOR_BGR2GRAY)

        # canny
        Lowthre = 100
        Highthre = 150
        edges = cv2.Canny(gray, Lowthre, Highthre, apertureSize=3)
        cv2.imwrite('./images/cannyimages/' + str(Lowthre) + ',' + str(Highthre) + ',' + str(i) + ".jpg", edges)

        # detect peek
        h, w = img270.shape[:2]
        cnts = countBlack(edges, tw=2)

        divideBorder = 8
        divideLines2 = []
        divw = 4

        for x in range(w - divw - 1):
            # pprint.pprint(cnts[x])
            if cnts[x] < divideBorder and all([cnts[tx] > divideBorder for tx in range(x + 1, x + divw)]):
                divideLines2.append(x)
        if len(divideLines2) is not None:
            divideLines2.insert(0, 0)
            divideLines2.append(w)

        fragmentPositions = []
        fragmentImages = []
        for y in range(0, len(divideLines2) - 1):
            # cv2.line(lineimg2, (divideLines2[y], 0), (divideLines2[y], h), (0, 0, 255), 10)
            divide_img = img270[0:h, divideLines2[y]:divideLines2[y + 1]]
            cv2.imwrite('./images/myhough2/' + str(divideLines2[y]) + ',' + str(stagePositions[i]) + ',' + str(divideLines2[y+1]) + ',' + str(stagePositions[i+1]) + ".jpg", divide_img)
            fragmentImages.append(divide_img)
            fragmentPos = {}
            fragmentPos['x1'] = divideLines2[y]
            fragmentPos['y1'] = stagePositions[i]
            fragmentPos['x2'] = divideLines2[y + 1]
            fragmentPos['y2'] = stagePositions[i + 1]
            fragmentPositions.append(fragmentPos)
        stageAndfragmentImages.append(fragmentImages)
        stageAndfragmentPositions.append(fragmentPositions)
        cv2.imwrite('./images/lineimages/' + str(Lowthre) + ',' + str(Highthre) + ',' + str(i) + ".jpg", lineimg2)
        f = open('stageAndfragmentImages_naname.pickle', 'wb')
        pickle.dump(stageAndfragmentImages, f)
        f = open('stageAndfragmentPositions_naname.pickle', 'wb')
        pickle.dump(stageAndfragmentPositions, f)
    # elapsed_time2 = time.time() - start2
    # print("elapsed_time2:{0}".format(elapsed_time2))
    return stageAndfragmentImages, stageAndfragmentPositions


GCV_OBJ2 = []
GOOGLE_CLOUD_VISION_API_URL = 'https://vision.googleapis.com/v1/images:annotate?key='
API_KEY = "AIzaSyDPv1cYnwwjxAlw4uy0aPhu9634sh7LxRM"
def ocrAndLevenshitein():
    start3 = time.time()

    LevenshteinDistanceBox2 = []
    LDAndtitle2Box = []
    stageAndfragmentImages, stageAndfragmentPositions = myHough2()
    for i, fragmentImage in enumerate(stageAndfragmentImages):
        LDAndtitle1Box = []
        LevenshteinDistanceBox1 = []
        for j, img in enumerate(fragmentImage):
            cv2.imwrite('image2.jpg', img)
            infile = open('image2.jpg', 'rb')
            stream = infile.read()
            infile.close()
            api_url = GOOGLE_CLOUD_VISION_API_URL + API_KEY
            start4 = time.time()
            req_body = json.dumps({
                'requests': [{
                    'image': {
                        'content': str(base64.b64encode(stream).decode("utf-8"))
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
            read_json = res.json()
            # elapsed_time4 = time.time() - start4
            # print("{0}".format(elapsed_time4))
            # pprint.pprint(res.json())

            # # JSONファイル書き出し
            # with open("./testJSON/naname/json" + str(i) + ";" + str(j) + ".json", 'w') as f:
            #     json.dump(read_json, f, sort_keys=True, indent=4)
            #
            # # JSONファイル読み込み
            # with open("./testJSON/naname/json" + str(i) + ";" + str(j) + ".json", 'r') as f:
            #     read_json = json.load(f)


                # pprint.pprint(read_json)
            # pprint.pprint(res.json())
            read_description = {}
            # start5 = time.time()
            if "textAnnotations" in read_json["responses"][0]:

                read_responses = read_json["responses"][0]
                read_textAnnotations = read_responses["textAnnotations"][0]
                read_description = read_textAnnotations["description"]

                # pprint.pprint(str(i) + ',' + str(j) + ',' + read_description)
                # if len(read_description) >= 2:

                LevenshteinDistanceTempBox = []
                LDAndtitleTempBox = []
                # for k, description, in enumerate(read_description):
                some_title = read_description.split("\n")
                some_title.pop(-1)
                #pprint.pprint(some_title)
                # some_title.append(description)  # 複数タイトルがある場合区切る
                MinValue = 400
                MinValueKey = -1
                for k, title, in enumerate(some_title):
                    #LDAndtitleTempBox.append({"distance_": Levenshtein.distance(title, target_title), "title": title})
                    LevenshteinDistanceTempBox.append(Levenshtein.distance(title, target_title))
                    if Levenshtein.distance(title, target_title) < MinValue:
                        MinValue = Levenshtein.distance(title, target_title)
                        MinValueKey = k
                LDAndtitle1Box.append({"distance_": MinValue, "title": some_title[MinValueKey]})
                # LevenshteinDistanceBox1.append(min(LevenshteinDistanceTempBox))


            else:
                LDAndtitle1Box.append({"distance_": 400, "title": "null"})
                # LevenshteinDistanceBox1.append(400)
            # elapsed_time5 = time.time() - start5
            # print("{0}".format(elapsed_time5))
        LDAndtitle2Box.append(LDAndtitle1Box)

        # LevenshteinDistanceBox2.append(LevenshteinDistanceBox1)

    # pprint.pprint("XXX")
    # pprint.pprint(LDAndtitle2Box)
    # pprint.pprint(LevenshteinDistanceBox2)

    LDAndtitlelist = []
    pointlist = []
    distancelist = []
    titlelist = []
    # start6 = time.time()
    for i, dan in enumerate(LDAndtitle2Box):
        for j, item in enumerate(dan):
            d = LDAndtitle2Box[i][j]["distance_"]
            # pprint.pprint(d)
            t = LDAndtitle2Box[i][j]["title"]
            LDAndtitlelist.append({"i": i, "j": j, "distance": d, "title": t})
    LDAndtitlelist.sort(key=lambda x: x["distance"])
    for k in range(6):
        stage = LDAndtitlelist[k]["i"]
        item = LDAndtitlelist[k]["j"]
        distancelist.append(LDAndtitlelist[k]["distance"])
        pointlist.append(stageAndfragmentPositions[stage][item])
        titlelist.append(LDAndtitlelist[k]["title"])
    # pprint.pprint(pointlist)
    # pprint.pprint(distancelist)
    # pprint.pprint(titlelist)
    for l in range(len(pointlist) - 1):
        x1 = pointlist[l]["x1"]
        y1 = pointlist[l]["y1"]
        x2 = pointlist[l]["x2"]
        y2 = pointlist[l]['y2']
        img = cv2.imread(entry_image)
        surroundedimage = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 16)
        cv2.imwrite('./images/surroundedimage/T' + str(l) + ".jpg", surroundedimage)
    # elapsed_time6 = time.time() - start6
    # print("elapsed_time6:{0}".format(elapsed_time6))
    # LDlist = []
    # pointlist = []
    # distancelist = []
    # for i, dan in enumerate(LevenshteinDistanceBox2):
    #     for j, item in enumerate(dan):
    #         LDlist.append({"i": i, "j": j, "distance": LevenshteinDistanceBox2[i][j]})
    # LDlist.sort(key=lambda x: x["distance"])
    # for k in range(6):
    #     stage = LDlist[k]["i"]
    #     item = LDlist[k]["j"]
    #     distancelist.append(LDlist[k]["distance"])
    #     pointlist.append(stageAndfragmentPositions[stage][item])
    # # pprint.pprint(pointlist)
    # # pprint.pprint(distancelist)
    # for l in range(len(pointlist) - 1):
    #     x1 = pointlist[l]["x1"]
    #     y1 = pointlist[l]["y1"]
    #     x2 = pointlist[l]["x2"]
    #     y2 = pointlist[l]['y2']
    #     img = cv2.imread(entry_image)
    #     surroundedimage = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 6)
    #     cv2.imwrite('./images/surroundedimage/' + str(l) + ".jpg", surroundedimage)
ocrAndLevenshitein()







