import cv2
import numpy as np
import mediapipe as mp
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cap0 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

pattern_size = (9, 6)   
square_size = 23.0       

# --- キャリブレーション読み込み ---
with open('stereo_calibration_data.pkl', 'rb') as f:
    calib = pickle.load(f)

camMtx1, dist1 = calib['cameraMatrix1'], calib['distCoeffs1']
camMtx2, dist2 = calib['cameraMatrix2'], calib['distCoeffs2']
R, T = calib['R'], calib['T']
R1, R2 = calib['R1'], calib['R2']

# --- グローバル変数として初期化 ---
fig = None
ax = None
scatter_tri = None
scatter_obj = None

criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def debug_chess_triangulate(frameL, frameR):
    """
    frameL, frameR: 左右画像（BGR）
    実行すると検出→正規化→三角測量→各種誤差出力→3D可視化
    """
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, pattern_size, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, pattern_size, None)

    print(cornersL)
    print(cornersR)

    if not (retL and retR):
        print("チェスボード検出不可")
        return

    # サブピクセル精度
    cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria_subpix)
    cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria_subpix)

    N = cornersL.shape[0]
    # object points
    objp = np.zeros((N, 3), np.float32)
    xs, ys = np.meshgrid(range(pattern_size[0]), range(pattern_size[1]))
    pts_idx = np.vstack([xs.ravel(), ys.ravel()]).T  # (N,2)
    objp[:, :2] = pts_idx * square_size               #実世界のサイズ(23mm)

    #歪み補正
    ptsL = cv2.undistortPoints(cornersL.reshape(-1,1,2).astype(np.float64),
                                    camMtx1, dist1, P=None).reshape(-1,2)
    ptsR = cv2.undistortPoints(cornersR.reshape(-1,1,2).astype(np.float64),
                                    camMtx2, dist2, P=None).reshape(-1,2)

    #三角推量
    P1 = np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = np.hstack((R, T.reshape(3,1)))
    pts4 = cv2.triangulatePoints(P1, P2, ptsL.T.astype(np.float64), ptsR.T.astype(np.float64))
    pts3d = (pts4[:3] / pts4[3]).T   # (N,3) : in camera1 coordinate system

    global fig
    global ax
    global scatter_tri
    global scatter_obj

    if fig is None:
        plt.ion()
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        scatter_tri = ax.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2],c='r')
        ax.set_title('chess board points')
        # --- 軸範囲を固定（必要に応じて調整）
        ax.set_xlim(-250, 250)
        ax.set_ylim(-250, 250)
        ax.set_zlim(200, 700)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()
    else:
        scatter_tri._offsets3d = (pts3d[:,0], pts3d[:,1], pts3d[:,2])
        plt.pause(0.001)  # 描画更新

while True:
    ret0, frame0 = cap0.read()
    ret2, frame2 = cap2.read()
    if not ret0 or not ret2:
        break

    debug_chess_triangulate(frame0.copy(), frame2.copy())

    cv2.imshow('cam0', frame0)
    cv2.imshow('cam2', frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap2.release()
cv2.destroyAllWindows()
plt.ioff()