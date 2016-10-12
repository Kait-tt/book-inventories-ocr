# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2

if __name__ == '__main__':
    
     # 画像の読み込み
    canny = cv2.imread("C:/image/book.jpg", 0)
    gray = cv2.imread("C:/image/book.jpg", 0)
    laplacian = cv2.imread("C:/image/book.jpg", 0)
    # エッジの抽出
    edge1 = cv2.Canny(canny, 50, 150)
    
    #Sobelフィルタでx方向のエッジ検出
    gray_sobelx = cv2.Sobel(gray,cv2.CV_32F,1,0)
    
    #Sobelフィルタでy方向のエッジ検出
    gray_sobely = cv2.Sobel(gray,cv2.CV_32F,0,1)

    #8ビット符号なし整数変換
    gray_abs_sobelx = cv2.convertScaleAbs(gray_sobelx) 
    gray_abs_sobely = cv2.convertScaleAbs(gray_sobely)
    
    #重み付き和
    gray_sobel_edge = cv2.addWeighted(gray_abs_sobelx,0.5,gray_abs_sobely,0.5,0)

    #ファイル保存
    cv2.imwrite("C:/image/edge1.jpg",edge1)
    cv2.imwrite("C:/image/gray_sobel_edge.jpg",gray_sobel_edge)
    
    # 結果表示(canny)
    cv2.imshow("Show Image",edge1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 結果表示(sobel)
    cv2.imshow('gray_sobel_edge',gray_sobel_edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    
