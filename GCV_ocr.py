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

GCV_OBJ = []
GOOGLE_CLOUD_VISION_API_URL = 'https://vision.googleapis.com/v1/images:annotate?key='
API_KEY = "AIzaSyDPv1cYnwwjxAlw4uy0aPhu9634sh7LxRM"
def google_cloud_vison():
    MY_HOUGH2 = "./images/myhough2"
    for file in glob.glob(os.path.join(MY_HOUGH2, "*.jpg")):
        print(file)
        stream = open(file, 'rb').read()
        # base64でencode
        infile = base64.b64encode(stream)
        # base64.b64encode(infile).decode("utf-8")
        # image_path, ext = os.path.splitext(os.path.basename(infile))

        api_url = GOOGLE_CLOUD_VISION_API_URL + API_KEY
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
        GCV_OBJ.append(res.json())
    pprint.pprint(GCV_OBJ)
    return res.json()
google_cloud_vison()

        # # リクエスト実行
        # # インスタンス生成
        # c = pycurl.Curl()
        # c.setopt(pycurl.URL, "https://vision.googleapis.com/v1/images:annotate?key=" + API_KEY)
        # # HTTPヘッダ取得
        # c.setopt(pycurl.HEADER, 1)
        # c.setopt(pycurl.CUSTOMREQUEST, 'POST')
        # c.setopt(pycurl.HTTPHEADER, ["Content-Type: application/json"])
        # c.setopt(pycurl.SSL_VERIFYPEER, False)
        # c.setopt(pycurl.TIMEOUT, 15)
        # c.setopt(pycurl.POSTFIELDS, req_body)
        # res1 = c.exec()
        # res2 = c.getinfo(pycurl.HEADER_SIZE)
        # c.close()
        # substr = []
        # req_body = substr(res1, res2)
        # header = substr(res1, 0, res2)



