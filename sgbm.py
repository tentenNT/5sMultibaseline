import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

np.savez("./calibrationResult0_"+str(right_camera_num)+".npz", rms=rms, mtxL2=cameraMatrix1, distL2=distCoeffs1, mtxR2=cameraMatrix2, distR2=distCoeffs2, R=R, T=T, E=E, F=F)

with np.load("./calibrationResult0_"+str(right_camera_num)+".npz") as X:
    rms, mtxL2, distL2, mtxR2, distR2, R, T, E, F = [X[i] for i in ("rms", "mtxL2", "distL2", "mtxR2", "distR2", "R", "T", "E", "F")]


