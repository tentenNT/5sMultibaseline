# From https://qiita.com/a2kiti/items/38171e6842b6332bba7b
# CHESS_SIZEは34.2っぽい
# 実行時の位置を基準にしているので他のディレクトリで実行するとそのディレクトリが汚染される
import numpy as np
import cv2
import sys
from time import sleep
import pdb

cv2.startWindowThread()

while True:
    cam_num = input ("Enter camera number: ")
    try:
        cap = cv2.VideoCapture(int(cam_num))
    except ValueError:
        print("Please retry")
        continue
    break

if cap.isOpened() is False:
    raise("IO Error")

# FPSの取得
fps = cap.get(cv2.CAP_PROP_FPS)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # カメラ画像の横幅を640に設定
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) # カメラ画像の縦幅を360に設定

# termination criteria
# [criteriaの指定]
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# [object point（三次元点の位置）とimage point（二次元画像上の点の位置）用の行列生成？]
objp = np.zeros((7*10,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
# [上の処理の続き]
objPoints = [] # 3d point in real world space
imgPoints = [] # 2d points in image plane.

imgInd=0
time_counter=0
while True:
    time_counter += 1
    ret, img = cap.read()
    if ret == False:
        continue
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#     cv2.putText(img,'Number of capture: '+str(imgInd),(30,20),cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0))
#     cv2.putText(img,'c: Capture the image',(30,40),cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0))
#     cv2.putText(img,'q: Finish capturing and calcurate the camera matrix and distortion',(30,60),cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0))
    cv2.imshow("image"+str(cam_num), img) 

    key = cv2.waitKey(1) & 0xFF

    #    if key == ord('c'):
    # 約2秒毎に撮影（ピッタリでは無い気がする）
    if time_counter % (int(fps)*2) == 0:

    # Find the chess board corners
    # [キャリブレーションボードからのコーナー決定]
        ret, corners = cv2.findChessboardCorners(gray, (10,7),None)

    # If found, add object points, image points (after refining them)
    # [コーナーが見つかったらobject pointsとimage pointsに入れておく]
        if ret == True:
            objPoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgPoints.append(corners2)

            # Draw and display the corners
            # [コーナーをディスプレイ上に表示]
            img = cv2.drawChessboardCorners(img, (10,7), corners2,ret)
            cv2.imshow('image'+str(cam_num),img)
            cv2.waitKey(500)
            cv2.imwrite('./data/calibration'+str(cam_num)+'/image'+str(imgInd) + 'Cam' + str(cam_num) + '.jpg', img)
            imgInd+=1
    # qキーか20枚撮影したら終了
    if key == ord('q') or imgInd == 20:
        break

# Calc urate the camera matrix
# [キャリブレーションパラメータの生成]
# [rms: 最終的な再投影誤差]
# [mtx: 内部パラメータ]
# [dist: レンズ歪みパラメータ]
# [rvecs: 回転ベクトル]
# [tvecs: 並進ベクトル]
try:
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1],None,None)
# Save the csv file
# [csvファイルに保存]
# だったのをnp.save()に置き換え
    np.savez("./cameraParams"+str(cam_num)+".npz", rms=rms, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs, objPoints=objPoints, imgPoints=imgPoints)
    # np.savetxt("mtx_cam" + str(cam_num) + ".csv", mtx, delimiter=",")
    # np.savetxt("dist_cam" + str(cam_num) + ".csv", dist, delimiter=",")
except cv2.error:
    cap.release()
    cv2.destroyAllWindows()
    print("calibration error!")
    sys.exit()

cap.release()
cv2.destroyAllWindows()
