# USBカメラ複数台を使った静止画撮影
# 今は同じカメラでしか使えない(FPSを個別に取得していない）
# カメラナンバーではなくリストのインデックスで取得している問題

import numpy as np
import cv2
import sys

cv2.startWindowThread()

# カメラの台数とcv2.VideoCapture()の受け皿を作っておく
cam_num = []
cam_list = []

while True:
    color_setting = input ("Enter color setting(r or g): ")

# cam_numを複数の数値入力に対応させる(https://qiita.com/863/items/b970d2f376c1e16c921b)
# うまくいかなかった
# print("Enter camera number(separator is space): ")
# cam_num.append((int(x) for x in input().split()))
# 代わりにカメラの台数を入力してrangeで取っているが微妙…直したい
    x = int(input("Enter number of cameras: "))
    cam_num = list(range(x))
    capture_time = input("Enter wait time(int only): ")
    img_name = input("Enter file name: ")
    try:
        for x in cam_num:
            cam_list.append(cv2.VideoCapture(x))
        capture_time = int(capture_time)
    except ValueError:
        print("Please retry")
        continue
            
    break

if cam_list[0].isOpened() is False:
    raise("IO Error")


# FPSの取得
fps = cam_list[0].get(cv2.CAP_PROP_FPS)

for video_input in cam_list:
    video_input.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # カメラ画像の横幅を640に設定
    video_input.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) # カメラ画像の縦幅を360に設定

imgInd=0
time_counter=0
imgInd_old=0

while True:
    while True:
        imgInd_old = imgInd
        time_counter += 1
        for ind, cl in enumerate(cam_list):
            ret, img = cl.read()
            if ret == False:
                continue
            if color_setting == "g":
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            cv2.imshow("Image"+str(ind), img) 

            # 指定された時間で撮影
            # int()で丸め込んでる
            if time_counter % (int(fps)*capture_time) == 0:
                cv2.imshow('Image'+str(ind), img)
                cv2.waitKey(500)
                cv2.imwrite('./data/cap/' + str(img_name) + '_cam' + str(ind) + "_" +str(imgInd) + '.jpg', img)

                imgInd+=1

        key = cv2.waitKey(1) & 0xFF
        # qキーか1枚撮影で終了
        if key == ord('q') or imgInd - imgInd_old== len(cam_num):
            break
    cv2.destroyAllWindows()
    frag = input("recapture?{} :".format(imgInd))
    if frag == "r":
        continue
    else:
        break

for cl in cam_list:
    cl.release()

