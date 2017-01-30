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

#entry_image = tkinter.filedialog.askopenfilename(title="Choose image file")
entry_image = './images/Book_Inventories.JPG'


# 高さ/3から2/3までの各列にいくつ白要素があるかカウントする
# tw: 列の幅
def countWhite(orig, tw=2):
    h, w = orig.shape[:2]
    minh = int(h / 3 * 1)
    maxh = int(h / 3 * 2)
    ary = orig.copy()[minh:maxh] == 255
    tmp = ary.copy()

    for i in range(tw - 1):
        tmp = np.delete(ary, 0, axis=1)
        ary = np.delete(ary, w - i - 1, axis=1)
        ary |= ary | tmp
    ary = np.delete(ary, w - tw, axis=1)
    cnts2 = np.sum(ary, axis=0)

    return cnts2


def myHough1(save_img=False):
    start1 = time.time()
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
    cnts = countWhite(edges, tw=2)
    # pprint.pprint(cnts)
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
        if save_img:
            cv2.line(lineimg1, (divideLines[x], 0), (divideLines[x], h), (0, 0, 255), 8)
            cv2.imwrite('./images/myhough1/' + str(x) + '.jpg', divide_img)

    if save_img:
        cv2.imwrite('./images/lineimages/' + str(Lowthre) + ',' + str(Highthre) + ',' + ".jpg", lineimg1)
    # print(np1)
    # elapsed_time1 = time.time() - start1
    # print("elapsed_time1:{0}".format(elapsed_time1))

    return divideImages, divideLines


def myHough2(save_img=False, save_pickle=False):
    # start2 = time.time()
    stageImages, stagePositions = myHough1(save_img=save_img)

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
        if save_img:
            cv2.imwrite('./images/cannyimages/' + str(Lowthre) + ',' + str(Highthre) + ',' + str(i) + ".jpg", edges)

        # detect peek
        h, w = img270.shape[:2]
        cnts = countWhite(edges, tw=2)

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
            cv2.imwrite('./images/myhough2/' + str(divideLines2[y]) + ',' + str(stagePositions[i]) + ',' + str(
                divideLines2[y + 1]) + ',' + str(stagePositions[i + 1]) + ".jpg", divide_img)
            fragmentImages.append(divide_img)
            fragmentPos = {}
            fragmentPos['x1'] = divideLines2[y]
            fragmentPos['y1'] = stagePositions[i]
            fragmentPos['x2'] = divideLines2[y + 1]
            fragmentPos['y2'] = stagePositions[i + 1]
            fragmentPositions.append(fragmentPos)

        stageAndfragmentImages.append(fragmentImages)
        stageAndfragmentPositions.append(fragmentPositions)
        if save_pickle:
            # TODO: 上書きして大丈夫なのか
            f = open('stageAndfragmentImages_naname.pickle', 'wb')
            pickle.dump(stageAndfragmentImages, f)
            f = open('stageAndfragmentPositions_naname.pickle', 'wb')
            pickle.dump(stageAndfragmentPositions, f)

        if save_img:
            cv2.imwrite('./images/lineimages/' + str(Lowthre) + ',' + str(Highthre) + ',' + str(i) + ".jpg", lineimg2)

    # elapsed_time2 = time.time() - start2
    # print("elapsed_time2:{0}".format(elapsed_time2))
    return stageAndfragmentImages, stageAndfragmentPositions


GCV_OBJ2 = []
GOOGLE_CLOUD_VISION_API_URL = 'https://vision.googleapis.com/v1/images:annotate?key='
API_KEY = "AIzaSyDPv1cYnwwjxAlw4uy0aPhu9634sh7LxRM"


def ocrAndLevenshitein(save_img=False):
    start3 = time.time()

    stageAndfragmentImages, stageAndfragmentPositions = myHough2(save_img)
    print("end hough: {:.4}".format(time.time() - start3))

    # 全ての本画像をBase64にエンコードし、リクエストオブジェクトを作る
    start4 = time.time()
    stage_idxes = []
    items_idxes = []
    request_objects = []
    for i, fragmentImage in enumerate(stageAndfragmentImages):
        for j, img in enumerate(fragmentImage):
            stage_idxes.append(i)
            items_idxes.append(j)
            cv2.imwrite('image2.jpg', img)
            infile = open('image2.jpg', 'rb')
            stream = infile.read()
            infile.close()
            base64image = str(base64.b64encode(stream).decode("utf-8"))
            request_objects.append({
                'image': {
                    'content': base64image
                },
                'features': [{
                    'type': 'TEXT_DETECTION',
                    'maxResults': 1
                }],
                'imageContext': {
                    'languageHints': ['ja']
                }
            })

    print("end base64 encoding: {:.4}".format(time.time() - start4))

    # 画像をできるだけまとめてAPIに送る
    start5 = time.time()
    # https://cloud.google.com/vision/limits
    LIMIT_BYTES = int(8 * 1024 * 1024 / 2)  # Max: 8MB
    LIMIT_BATCH = 16  # Max: 16
    api_url = GOOGLE_CLOUD_VISION_API_URL + API_KEY
    responses = []
    send_request_objects = []
    pos = 0
    while pos < len(request_objects):
        tmp_send_request_objects = send_request_objects.copy()
        tmp_send_request_objects.append(request_objects[pos])
        req_body = json.dumps({'requests': tmp_send_request_objects})
        if len(req_body) > LIMIT_BYTES or len(tmp_send_request_objects) > LIMIT_BATCH:
            if len(send_request_objects) == 0:  # 画像が大きすぎて送れない
                raise Exception('{}s image is too large.'.format(pos))

            req_body = json.dumps({'requests': send_request_objects})
            # print('request {} images'.format(len(send_request_objects)))
            res = requests.post(api_url, data=req_body)
            read_json = res.json()
            # read_json = {'responses': [{} for i in range(len(send_request_objects))]}  # stub
            responses.extend(read_json['responses'])
            send_request_objects = []
        else:
            send_request_objects = tmp_send_request_objects
            pos += 1

    if len(send_request_objects) > 0:
        req_body = json.dumps({'requests': send_request_objects})
        # print('request {} images'.format(len(send_request_objects)))
        res = requests.post(api_url, data=req_body)
        read_json = res.json()
        # read_json = {'responses': [{} for i in range(len(send_request_objects))]}  # stub
        responses.extend(read_json['responses'])

    print("end api request: {:.4}".format(time.time() - start5))

    # OCRの結果を取り出して編集距離を計算する
    start6 = time.time()
    LDAndtitleBox = []
    for i, response in enumerate(responses):
        if "textAnnotations" in response:
            read_textAnnotations = response["textAnnotations"][0]
            read_description = read_textAnnotations["description"]
            LevenshteinDistanceTempBox = []
            some_title = read_description.split("\n")
            some_title.pop(-1)
            # some_title.append(description)  # 複数タイトルがある場合区切る
            MinValue = 400
            MinValueKey = -1
            for k, title, in enumerate(some_title):
                LevenshteinDistanceTempBox.append(Levenshtein.distance(title, target_title))
                if Levenshtein.distance(title, target_title) < MinValue:
                    MinValue = Levenshtein.distance(title, target_title)
                    MinValueKey = k
            LDAndtitleBox.append({"stage": stage_idxes[i], "item": items_idxes[i],
                                  "distance": MinValue, "title": some_title[MinValueKey]})

        else:
            LDAndtitleBox.append({"stage": stage_idxes[i], "item": items_idxes[i],
                                  "distance": 400, "title": "null"})

    # pprint.pprint("XXX")
    # pprint.pprint(LDAndtitle2Box)
    # pprint.pprint(LevenshteinDistanceBox2)

    # 上からいくつか抽出
    # start6 = time.time()
    pointlist = []
    distancelist = []
    titlelist = []
    LDAndtitleBox.sort(key=lambda x: x["distance"])
    for k in range(6):
        stage = LDAndtitleBox[k]["stage"]
        item = LDAndtitleBox[k]["item"]
        distancelist.append(LDAndtitleBox[k]["distance"])
        pointlist.append(stageAndfragmentPositions[stage][item])
        titlelist.append(LDAndtitleBox[k]["title"])
    # pprint.pprint(pointlist)
    # pprint.pprint(distancelist)
    # pprint.pprint(titlelist)

    print("end Levenshtein: {:.4}".format(time.time() - start6))

    # 結果の画像を生成
    start7 = time.time()
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

    print("end Result Image: {:.4}".format(time.time() - start7))

    print("end All: {:.4}".format(time.time() - start3))

ocrAndLevenshitein()

# myHough2()
