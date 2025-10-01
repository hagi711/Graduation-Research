import numpy as np
import cv2 as cv
import glob
import pickle
import sys

# --- キャリブレーションマップ読み込み ---
with open('stereo_calibration_data.pkl', 'rb') as f:
    data = pickle.load(f)

map0 = data['map0']
map2 = data['map2']

cap0 = cv.VideoCapture(0)
cap2 = cv.VideoCapture(2)

while True:
    ret0,frame0 = cap0.read()
    ret2,frame2 = cap2.read()
    
    img2_und = cv.remap(frame2,map2[0],map2[1],interpolation=cv.INTER_LANCZOS4)
    img0_und = cv.remap(frame0,map0[0],map0[1],interpolation=cv.INTER_LANCZOS4)

    cv.imshow('frame0undist',img0_und)

    cv.imshow('frame2undist',img2_und)

    cv.imshow('frame0',frame0)

    cv.imshow('frame2',frame2)

    k = cv.waitKey(5) & 0xFF
    if k == ord('q') or k == 27:
        break

cap0.release()
cap2.release()
cv.destroyAllWindows()
