import numpy as np
import cv2 as cv
import glob
import pickle
import sys

cap0 = cv.VideoCapture(0)
cap2 = cv.VideoCapture(2)

# --- キャリブレーション読み込み ---
with open('calibration_data.pkl', 'rb') as f:
    calib = pickle.load(f)

cameraMat0, cameraDist0 = calib['cameraMatrix1'], calib['distCoeffs1']
cameraMat2, cameraDist2 = calib['cameraMatrix2'], calib['distCoeffs2']

# チェスボードの大きさ
chessboardSize = (9,6)

#　画像サイズ
framesize = np.array([640,480])

# チェスボード座標をグリッド座標へ変換
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

# チェスボードの１マスあたりの実世界での大きさ(mm)
size_of_chessboard_squares_mm = 23
objp = objp * size_of_chessboard_squares_mm

# 終了条件(繰り返し回数、誤差の大きさ)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)



# 世界座標と画像座標
objpoints = []  # 世界座標（出力）
imgpoints0 = []  # カメラ０の画像座標（入力）
imgpoints2 = [] # カメラ２の画像座標（入力）

# 回転行列
R = np.array((3,3),np.float32)

# 並進行列
T = np.array((3,1),np.float32)

# それぞれの画像の読み込み
images0 = sorted(glob.glob('./images/camera0/*.png'))
images2 = sorted(glob.glob('./images/camera2/*.png'))

# 一致しているか確認
assert len(images0) == len(images2), "左右の画像枚数が一致しません"

for img0_path, img2_path in zip(images0, images2):
    img0 = cv.imread(img0_path)
    img2 = cv.imread(img2_path)
    gray0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    ret0, corners0 = cv.findChessboardCorners(gray0, chessboardSize, None)
    ret2, corners2 = cv.findChessboardCorners(gray2, chessboardSize, None)

    if ret0 and ret2:
        corners0 = cv.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria).reshape(-1, 2)
        corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria).reshape(-1, 2)

        objpoints.append(objp)         # 1ペアごとに追加
        imgpoints0.append(corners0)
        imgpoints2.append(corners2)

        '''
        cv.drawChessboardCorners(img0, chessboardSize, corners0, ret0)
        cv.imshow('img0', img0)

        cv.drawChessboardCorners(img2, chessboardSize, corners2, ret2)
        cv.imshow('img2', img2)
        cv.waitKey(500)
        '''
        

        


cv.destroyAllWindows()

flags = cv.CALIB_USE_INTRINSIC_GUESS

print('objlen')
print(len(objpoints))   # → 画像の枚数ぶんある？
print('img0len')
print(len(imgpoints0))   # → 左画像の検出点（枚数分）
print('img2len')
print(len(imgpoints2))   # → 右画像の検出点（枚数分）
print('objshape')
print(objpoints[0].shape)   # → (N, 3)
print('img0shape')
print(imgpoints0[0].shape)   # → (N, 2)
print('img2shape')
print(imgpoints2[0].shape)

# ステレオキャリブレーション
ret, _, _, _, _, R, T, _, _ = cv.stereoCalibrate(
    objpoints, imgpoints0, imgpoints2,
    cameraMat0, cameraDist0,
    cameraMat2, cameraDist2,
    framesize,
    criteria=criteria,
    flags=cv.CALIB_USE_INTRINSIC_GUESS
)
print(T)
print(R)

R1,R2 = np.array((3,3),np.float32)
P1,P2 = np.array((3,4),np.float32)
Q = np.array((4,4),np.float32)



R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(cameraMat0,cameraDist0,cameraMat2,cameraDist2,framesize,R,T)

map0 = cv.initUndistortRectifyMap(cameraMat0,cameraDist0,R1,P1,framesize,cv.CV_32FC1)

map2 = cv.initUndistortRectifyMap(cameraMat2,cameraDist2,R2,P2,framesize,cv.CV_32FC1)

# 保存ファイル名
filename = 'stereo_calibration_data.pkl'

# 保存する辞書
data = {
    'cameraMatrix1':cameraMat0,
    'cameraMatrix2':cameraMat2,
    'distCoeffs1':cameraDist0,
    'distCoeffs2':cameraDist2,
    'R': R,
    'T': T,
    'R1': R1,
    'R2': R2,
    'P1': P1,
    'P2': P2,
    'Q': Q,
    'map0': map0,
    'map2': map2
}

# ファイルに保存
with open(filename, 'wb') as f:
    pickle.dump(data, f)

print("キャリブレーションデータを保存しました")


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


