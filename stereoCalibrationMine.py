# -*- coding: utf-8 -*-
""" Created on Tue Jul 14 20:10:21 2015

@author: SANSON
"""

import numpy as np
import cv2
from glob import glob
import tkinter as Tkinter
from tkinter import messagebox as tkMessageBox

cv2.startWindowThread()
left_camera_num = int(input("left_camera_num: "))
right_camera_num = int(input("right_camera_num: "))

#--------------------------------------------------------1.カメラそれぞれのキャリブレーション
square_size = 50.0      # 正方形のサイズ
pattern_size = (9, 14)  # 格子数
pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 ) #チェスボード（X,Y,Z）座標の指定 (Z=0)
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
obj_points = []
img_points = []

#-----------------------------------------------1-1.左カメラ（基準カメラ）
for fn in glob("./data/cal_0/calibration" + str(left_camera_num) + "/*.jpg"):
    # 画像の取得
    im = cv2.imread(fn, 0)
    print("loading..." + fn)
    # チェスボードのコーナーを検出
    found, corner = cv2.findChessboardCorners(im, pattern_size)
    # コーナーがあれば
    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(im, corner, (11,11), (-1,-1), term)    #サブピクセル処理（小数点以下のピクセル単位まで精度を求める）
        cv2.drawChessboardCorners(im, pattern_size, corner,found)
        cv2.waitKey(100)
        cv2.imshow('found corners in ' + fn,im)
        cv2.waitKey(100)
    # コーナーがない場合のエラー処理
    if not found:
        print('chessboard not found')
        continue
    # 選択ボタンを表示
    root = Tkinter.Tk()
    root.withdraw()
    if tkMessageBox.askyesno('askyesno','この画像の値を採用しますか？'):
         img_points.append(corner.reshape(-1, 2))   #appendメソッド：リストの最後に因数のオブジェクトを追加 #corner.reshape(-1, 2) : 検出したコーナーの画像内座標値(x, y)
         obj_points.append(pattern_points)
         print('found corners in ' + fn + ' is adopted')
    else:
         print('found corners in ' + fn + ' is not adopted')
    cv2.destroyAllWindows()
    
    
# 内部パラメータを計算
rms, mtx_l, d_l, r, t = cv2.calibrateCamera(obj_points,img_points,(im.shape[1],im.shape[0]), None, None)
# 計算結果を表示
print("RMS_l = ", rms)
print("mtx_l = \n", mtx_l)
print("d_l = ", d_l.ravel())
# 計算結果を保存
np.savetxt("mtx_left.csv", mtx_l, delimiter =',',fmt="%0.14f") #カメラ行列の保存
np.savetxt("d_left.csv", d_l, delimiter =',',fmt="%0.14f") #歪み係数の保存
np.savez("./calibrationResultL.npz", rms=rms, mtx=mtx_l, dist=d_l, rvecs=r, tvecs=t, objPoints=obj_points, imgPoints=img_points)

#--------------------------------------------------------1-2.右カメラ

obj_points = []
img_points = []


for fn in glob("./data/cal_0/calibration" + str(right_camera_num) + "/*.jpg"):
    # 画像の取得
    im = cv2.imread(fn, 0)
    print("loading..." + fn)
    # チェスボードのコーナーを検出
    found, corner = cv2.findChessboardCorners(im, pattern_size)
    # コーナーがあれば
    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(im, corner, (11,11), (-1,-1), term)    #サブピクセル処理（小数点以下のピクセル単位まで精度を求める）
        cv2.drawChessboardCorners(im, pattern_size, corner,found)
        cv2.waitKey(100)
        cv2.imshow('found corners in ' + fn,im)
        cv2.waitKey(100)
    # コーナーがない場合のエラー処理
    if not found:
        print('chessboard not found')
        continue
    # 選択ボタンを表示
    root = Tkinter.Tk()
    root.withdraw()
    if tkMessageBox.askyesno('askyesno','この画像の値を採用しますか？'):
         img_points.append(corner.reshape(-1, 2))   #appendメソッド：リストの最後に因数のオブジェクトを追加 #corner.reshape(-1, 2) : 検出したコーナーの画像内座標値(x, y)
         obj_points.append(pattern_points)
         print('found corners in ' + fn + ' is adopted')
    else:
         print('found corners in ' + fn + ' is not adopted')
    cv2.destroyAllWindows()
    
    
# 内部パラメータを計算
rms, mtx_r, d_r, r, t = cv2.calibrateCamera(obj_points,img_points,(im.shape[1],im.shape[0]), None, None)
# 計算結果を表示
print("RMS_r = ", rms)
print("mtx_r = \n", mtx_r)
print("d_r = ", d_r.ravel())
# 計算結果を保存
np.savetxt("mtx_right.csv", mtx_r, delimiter =',',fmt="%0.14f") #カメラ行列の保存
np.savetxt("d_right.csv", d_r, delimiter =',',fmt="%0.14f") #歪み係数の保存
np.savez("./calibrationResultR.npz", rms=rms, mtx=mtx_r, dist=d_r, rvecs=r, tvecs=t, objPoints=obj_points, imgPoints=img_points)

#--------------------------------------------------------2.ステレオビジョンシステムのキャリブレーション
N = 18 #キャリブレーション用ステレオ画像のペア数
#「left0.jgp」のように、ペア番号を'left','right'の後につけて同じフォルダに置く(grobが使いこなせれば直したい)

square_size = 50.0      # 正方形のサイズ
pattern_size = (9, 14)  # 模様のサイズ
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32 ) #チェスボード（X,Y,Z）座標の指定 (Z=0)
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
obj_points = []
img_points1 = []
img_points2 = []

for i in range(N):
    # 画像の取得
    im_l = cv2.imread("./data/cal_0/calibration" + str(left_camera_num) + "/cal_cam" + str(left_camera_num) + "_"+str(left_camera_num + i*3)+ ".jpg", 0)
    im_r = cv2.imread("./data/cal_0/calibration" + str(right_camera_num) + "/cal_cam"+str(right_camera_num)+"_"+str(right_camera_num + i*3)+ ".jpg", 0)
    print("loading..." + "left" +str(i)+ ".jpg")
    print("loading..." + "right" +str(i)+ ".jpg")
    #コーナー検出
    found_l, corner_l = cv2.findChessboardCorners(im_l, pattern_size)
    found_r, corner_r = cv2.findChessboardCorners(im_r, pattern_size)
    # コーナーがあれば
    if found_l and found_r:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
        cv2.cornerSubPix(im_l, corner_l, (11,11), (-1,-1), term)
        cv2.cornerSubPix(im_r, corner_r, (11,11), (-1,-1), term)
        cv2.drawChessboardCorners(im_l, pattern_size, corner_l,found_l)
        cv2.drawChessboardCorners(im_r, pattern_size, corner_r,found_r)
        cv2.waitKey(100)
        cv2.imshow('found corners in ' + "left" +str(i)+ ".jpg", im_l)
        cv2.waitKey(100)
        cv2.imshow('found corners in ' + "right" +str(i)+ ".jpg", im_r)
        cv2.waitKey(100)
    # コーナーがない場合のエラー処理
    if not found_l:
        print('chessboard not found in leftCamera')
        continue
    if not found_r:
        print('chessboard not found in rightCamera')
        continue # 選択ボタンを表示 root = Tkinter.Tk() root.withdraw()
    if tkMessageBox.askyesno('askyesno','この画像の値を採用しますか？'):
         img_points1.append(corner_l.reshape(-1, 2))
         img_points2.append(corner_r.reshape(-1, 2))
         obj_points.append(pattern_points)
         print('found corners in ' + str(i) + ' is adopted')
    else:
         print('found corners in ' + str(i) + ' is not adopted')
    cv2.destroyAllWindows()

# システムの外部パラメータを計算
# imageSize = (im_l.shape[1],im_l.shape[0])
cameraMatrix1 = mtx_l
cameraMatrix2 = mtx_r
distCoeffs1 = d_l
distCoeffs2 = d_r
im_l = cv2.imread("./data/cal_0/calibration" + str(left_camera_num) + "/cal_cam" + str(left_camera_num) +"_" + str(left_camera_num) + ".jpg", 0)
h, w = im_l.shape[:2]
(retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F) = cv2.stereoCalibrate(obj_points, img_points1, img_points2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (w,h))

# 計算結果を表示
print("retval = ", retval)
print("R = \n", R)
print("T = \n", T)
# 計算結果を保存
np.savetxt("cameraMatrix" + str(left_camera_num) + ".csv", cameraMatrix1, delimiter =',',fmt="%0.14f") #新しいカメラ行列を保存
np.savetxt("cameraMatrix" + str(right_camera_num) + ".csv", cameraMatrix2, delimiter =',',fmt="%0.14f") 
np.savetxt("distCoeffs1.csv", distCoeffs1, delimiter =',',fmt="%0.14f") #新しい歪み係数を保存
np.savetxt("distCoeffs2.csv", distCoeffs2, delimiter =',',fmt="%0.14f")
np.savetxt("R" + str(left_camera_num) + "_" + str(right_camera_num) + ".csv", R, delimiter =',',fmt="%0.14f") #カメラ間回転行列の保存
np.savetxt("T" + str(left_camera_num) + "_" + str(right_camera_num) + ".csv", T, delimiter =',',fmt="%0.14f") #カメラ間並進ベクトルの保存
np.savez("./calibrationResult" + str(left_camera_num) +"_"+str(right_camera_num)+".npz", rms=rms, mtxL2=cameraMatrix1, distL2=distCoeffs1, mtxR2=cameraMatrix2, distR2=distCoeffs2, R=R, T=T, E=E, F=F)


#--------------------------------------------------------平行化変換以降は、「cv2.stereoRectify_and_Matching.py」へ
