import numpy as np
import cv2
import sys
import pdb

cv2.startWindowThread()
cam_list = []

while True:
    color_setting = input ("Enter color setting(r or g): ")
    cam_num = (0, 1, 2)
    try:
        for x in cam_num:
            cam_list.append(cv2.VideoCapture(x))
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
    for ind, cl in enumerate(cam_list):
        ret, img = cl.read()
        if ret == False:
            continue
        if color_setting == "g":
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow("Image"+str(ind), img) 
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()

for cl in cam_list:
    cl.release()

