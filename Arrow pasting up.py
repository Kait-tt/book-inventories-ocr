# -*- coding: utf-8 -*-
import Image

def paste():
    img = Image.open("./images/BI12_iso800tri.jpg")
    arrow = Image.open("./images/arrow.jpg")
    
    img.paste(arrow, (80, 0))
    img.save("SearchCompletion.jpg")
        
paste()