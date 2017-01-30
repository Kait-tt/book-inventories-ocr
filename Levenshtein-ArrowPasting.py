# -*- coding: utf-8 -*-

import Levenshtein
import linecache
from PIL import Image
import glob
import os
import sys
import tkinter


cnt_FileNumber = 0
input_lines1 = sys.stdin.readline()
OCR_folder_path = "C:/xampp/htdocs/json/cut_i"
FileTitle = []
read_title = []
LevenshteinDBox = []
min_idx = 0

for infile in glob.glob(os.path.join(OCR_folder_path, "*.text")):
    path, ext = os.path.splitext(infile)
    path, ext = os.path.splitext(os.path.basename(infile))
    FileTitle.append(path)
    target_line = linecache.getline(infile, 7) #文字認識の結果を読み取る
    read_title.append(target_line[26:])
print(read_title)
target_title = input_lines1
for x in range(0, len(read_title)):
    LevenshteinDBox.append(Levenshtein.distance(read_title[x], target_title))
print(LevenshteinDBox)
key = 0
key = LevenshteinDBox.index(min(LevenshteinDBox))
print(min(LevenshteinDBox))
X = -1
Y = -1
Str = ""
for i in range(0, len(FileTitle[key])):
    if FileTitle[key][i] == ",":
        X = int(Str)
        Str = ""
    else:
        Str += FileTitle[key][i]
Y = int(Str)

print(X)
print(Y)
        
#FileTitle[key]に座標(xxxx,yyyy)x,yは任意の数
        
def paste():
    img = Image.open("./images/BI12_iso800tri.jpg")
    #高さ,幅
    img.size[1:]
    arrow = Image.open("./images/arrow.png")
    img.paste(arrow, (Y - 130, img.size[1] - X))
    img.save("SearchCompletion.jpg")

paste()
