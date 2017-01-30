import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
from matplotlib.font_manager import FontProperties

# detect peek
img = cv2.imread('./images/BI12_iso800triro.jpg')
ary = img
tw = 2
h, w = ary.shape[:2]
minh = int(h / 3 * 1)
maxh = int(h / 3 * 2)
cnts = []
print(h,w)
for sx in range(0, w - tw):
    cnt = 0
    for y in range(minh, maxh):
        if any([ary[y][x] == 255 for x in range(sx, sx + tw)]):
            cnt += 1
    cnts.append(cnt)
fp = FontProperties(fname='C:\WINDOWS\Fonts\YuGothic.ttf')
plt.plot(range(0, len(cnts)), cnts)
plt.plot(w, [cnts[x] for x in w], 'o')
plt.xlim(0, 4072)
plt.xlabel(u'画像の幅(ピクセル)', fontproperties=fp)
plt.ylabel(u'黒の数(個)', fontproperties=fp)
plt.savefig("graph.png")