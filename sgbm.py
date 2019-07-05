import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob 
import csv
import pdb

cv2.startWindowThread()
left_camera_num = int(input("left_camera_num: "))
right_camera_num = int(input("right_camera_num: "))
image_num = int(input("image_num: "))
with np.load("./calibrationResult" + str(left_camera_num) +"_"+str(right_camera_num)+".npz") as X:
    rms, mtxL2, distL2, mtxR2, distR2, R, T, E, F = [X[i] for i in ("rms", "mtxL2", "distL2", "mtxR2", "distR2", "R", "T", "E", "F")]

# 0_1は常に読み込ませておく(test)
# with np.load("./calibrationResult0_1.npz") as X:
#     rms0_1, mtxL0_1, distL0_1, mtxR0_1, distR0_1, R0_1, T0_1, E0_1, F0_1 = [X[i] for i in ("rms", "mtxL2", "distL2", "mtxR2", "distR2", "R", "T", "E", "F")]

# left_cam_num == 1 かつright_cam_num == 2ならobj_pointsを平行移動させる

imgL = cv2.imread("./data/calibration" + str(left_camera_num) + "/cap_cam"+str(left_camera_num)+ "_"+ str(image_num*3) + ".jpg")
imgR = cv2.imread("./data/calibration"+str(right_camera_num)+"/cap_cam"+str(right_camera_num)+"_"+str(right_camera_num + image_num*3)+".jpg")
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

h, w = imgL.shape[:2]



# 平行化のための回転行列
flags = 0
alpha = 1
RpL, RpR, PpL, PpR, Q, validPixROI_L, validPixROI_R = \
    cv2.stereoRectify(mtxL2, distL2, mtxR2, distR2, (w,h), R, T, flags, alpha, (w,h))

# テスト
# RpL0_1, RpR0_1, PpL0_1, PpR0_1, Q0_1, validPixROI_L0_1, validPixROI_R0_1 = \
#     cv2.stereoRectify(mtxL0_1, distL0_1, mtxR0_1, distR0_1, (w,h), R0_1, T0_1, flags, alpha, (w,h))

m1type = cv2.CV_32FC1
map4pL = cv2.initUndistortRectifyMap(mtxL2, distL2, RpL, PpL, (w,h), cv2.CV_32FC1)
map4pR = cv2.initUndistortRectifyMap(mtxR2, distR2, RpR, PpR, (w,h), cv2.CV_32FC1)

rectifiedImgL = cv2.remap(imgL, map4pL[0], map4pL[1], cv2.INTER_LINEAR)
rectifiedImgR = cv2.remap(imgR, map4pR[0], map4pR[1], cv2.INTER_LINEAR)

cv2.waitKey(100)
cv2.imshow("rectifiedImgL", rectifiedImgL)
cv2.waitKey(100)
cv2.imshow("rectifiedImgR", rectifiedImgR)
cv2.waitKey(10000000)
cv2.destroyAllWindows()
cv2.imwrite("./data/rectifiedImgL.jpg", rectifiedImgL)
cv2.imwrite("./data/rectifiedImgR.jpg", rectifiedImgR)
windowSize = 1 # ブロックサイズ（小さめに設定する）
minDisp = 630   # 視差の下限（通常 0）
numDisp = 80   # 視差の個数の上限（最大視差＝ minDisp + numDisp）
"""
stereo = cv2.StereoSGBM_create(
        minDisparity = minDisp,      # 視差の下限
        numDisparities = numDisp,    # 視差の個数の上限
        P2 = 32*3*windowSize**2,     # 視差のなめらかさを制御するパラメータ2
        disp12MaxDiff = 1,           # left-right 視差チェックにおけて許容される最大の差
        uniquenessRatio = 30,        # マッチングの一意性のパラメータ（パーセント単位で表現）
        speckleWindowSize = 100,     # 視差計算時の連結成分の最大サイズ
        speckleRange = 32            # それぞれの連結成分における視差の最大差
    )
"""
stereo = cv2.StereoSGBM_create(
        minDisparity = minDisp,      # 視差の下限
        numDisparities = numDisp,    # 視差の個数の上限
        P2 = 32*3*windowSize**2,     # 視差のなめらかさを制御するパラメータ2
        disp12MaxDiff = 1,           # left-right 視差チェックにおけて許容される最大の差
        uniquenessRatio = 50,        # マッチングの一意性のパラメータ（パーセント単位で表現）
        preFilterCap = 63,
        speckleWindowSize = 250,     # 視差計算時の連結成分の最大サイズ
        speckleRange = 1,            # それぞれの連結成分における視差の最大差
        mode = cv2.STEREO_SGBM_MODE_HH
        # mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

rectifiedGrayL = cv2.cvtColor(rectifiedImgL, cv2.COLOR_BGR2GRAY)
rectifiedGrayR = cv2.cvtColor(rectifiedImgR, cv2.COLOR_BGR2GRAY)

disparity = stereo.compute(rectifiedGrayL, rectifiedGrayR).astype(np.float32) / 16
# disparity = (disparity - minDisp) / numDisp
# disparity = stereo.compute(rectifiedGrayL, rectifiedGrayR)
# ノーマライズ
# disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

fig = plt.figure(figsize=(16,9))
plt.imshow(disparity, cmap="gray")
plt.show()


# ply_header = '''ply
# format ascii 1.0
# element vertex %(vert_num)d
# property float x
# property float y
# property float z
# property uchar red
# property uchar green
# property uchar blue
# end_header
# '''

# ply形式の3Dモデルファイルを生成
def writeAsPly(outFilePath, verts, colors, left_camera_num, right_camera_num):
    verts = verts.reshape(-1, 3)
    # カメラ次第で追加の平行移動
    # if left_camera_num == 1 and right_camera_num == 2:
    #     print("追加で移動")
    #     verts[:,0] += T[0] - T0_1[0]
    #     verts[:,1] += T[1] - T0_1[1]
    #     verts[:,2] += T[2] - T0_1[2]
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors/255.0])
    # 不要な点群を除去
    # verts = np.delete(verts, np.where(verts[:,2] > 3500)[0], axis=0)
    
    with open(outFilePath, 'w', encoding='utf-8') as csvFile:
        writer = csv.writer(csvFile, delimiter=" ", lineterminator='\n')
        writer.writerows(ply_header)
        for vs in verts:
            writer.writerow(vs)

# XYZ
points = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=False)
# テスト
# points = cv2.reprojectImageTo3D(disparity, Q0_1, handleMissingValues=False)

# RGB
colors = cv2.cvtColor(rectifiedImgL,cv2.COLOR_BGR2RGB)

# 最小視差より大きな値を抽出
mask = disparity < minDisp
points[points==float('inf')] = 0
points[points==float('-inf')] = 0
# mask処理
outPoints = points#[mask]
outColors = colors#[mask]

ply_header = [['ply'], 
               ['format','ascii','1.0'],
               # 通常の点群取得用
               ['element', 'vertex', "{}".format(outPoints.shape[0]*outPoints.shape[1])],
               # 点群除去用（力技）
#                ['element', 'vertex', "40100"],
#                ['element', 'vertex', "56209"],
#                ['element', 'vertex', "17344"],
#                ['element', 'vertex', "22079"],
               ['property', 'float', 'x'],
               ['property', 'float', 'y'],
               ['property', 'float', 'z'],
               ['property', 'float', 'red'],
               ['property', 'float', 'green'],
               ['property', 'float', 'blue'],
               ['end_header']]

# print(outPoints.shape)
# print(np.max(outPoints))
# print(np.min(outPoints))
# print(disparity.shape)
# print(points.shape)

# 視差画像からx,y,z座標を取得
print('generating 3d point cloud...')

# plyファイルを生成
filename = "pointcloud" + str(left_camera_num) + "_" + str(right_camera_num) + ".ply"
outFilePath = os.path.join("./data", filename)

#write_ply(outFilePath, outPoints, outColors)
writeAsPly(outFilePath, outPoints, outColors, left_camera_num, right_camera_num)

print('finished')
