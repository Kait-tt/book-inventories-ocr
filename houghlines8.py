# -*- coding: utf-8 -*-
from PIL import Image
import cv2
import numpy as np
from matplotlib import interactive

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
    

def myhough():
    
    img = Image.open("./images/BI10.jpg")
    img90PIL = img.transpose(Image.ROTATE_90)
    img = cv2.cvtColor(np.array(img90PIL) , cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    # gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # canny
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)

    # detect peek
    cnts = countBlack(edges, tw=2)
    '''
    plt.plot(range(0, len(cnts)), cnts)
    plt.savefig('./images/.png')  
    plt.show()
    '''
    
    divideBorder = 8
    divideLines = []
    divw = 10
    for x in range(w - divw - 1):
        if cnts[x] < divideBorder and all([cnts[tx] > divideBorder for tx in range(x + 1, x + divw)]):
            divideLines.append(x)
            
    dLMm1 = len(divideLines) -1
    # draw lines
    for x in range(0,dLMm1):
        cv2.line(img, (divideLines[x], 0), (divideLines[x], h), (0, 0, 255), 2)
        dst = img[0:h,divideLines[x]:divideLines[x+1]]
        strx = str(divideLines[x])
        strh = str(h)                                                                        
        cv2.imwrite('./images/'+"BI10test_" + strx +","+ strh +".jpg" ,dst)
    '''
    cv2.imshow("divideimage", img)
    cv2.imwrite("./images/BI10_cut.jpg",img)
    '''

myhough()

cv2.waitKey(0)
cv2.destroyAllWindows()
