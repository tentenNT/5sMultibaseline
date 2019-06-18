# From https://qiita.com/a2kiti/items/38171e6842b6332bba7b
# CHESS_SIZEは34.2っぽい
# 保存したファイルの使い方は以下 
# mtx = np.loadtxt("mtx.csv",delimiter=",")
# dist = np.loadtxt("dist.csv",delimiter=",")
import numpy as np
import cv2
import sys
from time import sleep

# 引数をミリ秒単位で受け取るmsleep()を作っておく
# （使いませんでした）
import time
msleep = lambda x: time.sleep(x/1000.0)

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

# termination criteria
# [criteriaの指定]
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# [object point（三次元点の位置）とimage point（二次元画像上の点の位置）用の行列生成？]
objp = np.zeros((7*10,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
# [上の処理の続き]
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

imgInd=0
time_counter=0
while True:
    time_counter += 1
    ret, img = cap.read()
    if ret == False:
        continue
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    cv2.putText(img,'Number of capture: '+str(imgInd),(30,20),cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0))
    cv2.putText(img,'c: Capture the image',(30,40),cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0))
    cv2.putText(img,'q: Finish capturing and calcurate the camera matrix and distortion',(30,60),cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0))
    cv2.imshow("Image", img) 

    key = cv2.waitKey(1) & 0xFF

    # ここの処理を変える
    #    if key == ord('c'):
    # 約2秒毎に撮影（ピッタリでは無い気がする）
    if time_counter % (int(fps)*2) == 0:

    # Find the chess board corners
    # [キャリブレーションボードからのコーナー決定]
        ret, corners = cv2.findChessboardCorners(gray, (10,7),None)

    # If found, add object points, image points (after refining them)
    # [コーナーが見つかったらobject pointsとimage pointsに入れておく]
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            # [コーナーをディスプレイ上に表示]
            img = cv2.drawChessboardCorners(img, (10,7), corners2,ret)
            cv2.imshow('Image',img)
            cv2.waitKey(500)
            cv2.imwrite('./data/multiCalib/Image' + str(imgInd) + '_cam' + str(cam_num) + '.jpg', img)
            imgInd+=1
    # qキーか15枚撮影したら終了
    if key == ord('q') or imgInd == 15:
        break

# Calc urate the camera matrix
# [キャリブレーションパラメータの生成]
# [ret: ブール値]
# [mtx: 内部パラメータ]
# [dist: レンズ歪みパラメータ]
# [rvecs: 回転ベクトル]
# [tvecs: 並進ベクトル]
try:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
# Save the csv file
# [csvファイルに保存]
    np.savetxt("mtx_cam" + str(cam_num) + ".csv", mtx, delimiter=",")
    np.savetxt("dist_cam" + str(cam_num) + ".csv", dist, delimiter=",")
except cv2.error:
    cap.release()
    cv2.destroyAllWindows()
    print("calibration error!")
    sys.exit()


cap.release()
cv2.destroyAllWindows()

img = cv2.imread('Image0_cam' + str(cam_num) + '.jpg')
h, w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
# [歪み補正]
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
# [歪み補正したイメージ生成]
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('./data/multiCalibresult.png',dst)
