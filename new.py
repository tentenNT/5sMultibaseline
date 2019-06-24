import cv2
import numpy as np
import pdb

cam_num = 0
# rms, mtx, dist, rvecs, tvecs, objPoints, imgPoints = np.load("./cameraParams"+str(cam_num)+".npz")
with np.load("./cameraParams"+str(cam_num)+".npz") as X:
    rms, mtx, dist, rvecs, tvecs, objPoints, imgPoints = [X[i] for i in ('rms','mtx','dist','rvecs','tvecs', 'objPoints', 'imgPoints')]
img = cv2.imread('./data/calibration'+str(cam_num)+'/image0Cam'+str(cam_num)+'.jpg')
pdb.set_trace()
cv2.imshow("img",img)
h, w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
# [歪み補正]
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
# [歪み補正したイメージ生成]
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('./data/calibresult'+str(cam_num)+'.png',dst)
