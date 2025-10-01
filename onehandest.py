import cv2
import numpy as np
import mediapipe as mp
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS


# --- キャリブレーション読み込み ---
with open('stereo_calibration_data.pkl', 'rb') as f:
    calib = pickle.load(f)

camMtx1, dist1 = calib['cameraMatrix1'], calib['distCoeffs1']
camMtx2, dist2 = calib['cameraMatrix2'], calib['distCoeffs2']
P1, P2 = calib['P1'], calib['P2']
R1, R2 = calib['R1'], calib['R2']

# --- MediaPipe Hand ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

# --- カメラ準備 ---
cap0 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

# --- MediaPipe で手検出 ---
def get_landmarks(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if result.multi_hand_landmarks:
        return [(lm.x * img.shape[1], lm.y * img.shape[0]) for lm in result.multi_hand_landmarks[0].landmark]
    return None

def hand_estimate(img0,img2):
    landmarks0 = get_landmarks(img0)
    landmarks2 = get_landmarks(img2)
    print(landmarks0)

    print(landmarks0)
    print('-----------------------------------------------')
    print(landmarks2)

    if landmarks0 is None:
        print("カメラ0で手が検出できませんでした")
        return
    if landmarks2 is None:
        print("カメラ2で手が検出できませんでした")
        return

    # --- ランドマークを np.array に変換 ---
    pts0 = np.array(landmarks0, dtype=np.float32).reshape(-1, 1, 2)
    pts2 = np.array(landmarks2, dtype=np.float32).reshape(-1, 1, 2)

    # --- undistortPoints (補正前 → 補正後画像座標系へ変換) ---
    pts0_rect = cv2.undistortPoints(pts0, camMtx1, dist1, R=R1, P=P1)
    pts2_rect = cv2.undistortPoints(pts2, camMtx2, dist2, R=R2, P=P2)

    # --- 三角測量で3D座標復元 ---
    pts4d_hom = cv2.triangulatePoints(P1, P2, pts0_rect, pts2_rect)
    pts3d = (pts4d_hom[:3] / pts4d_hom[3]).T  # [N,3]
    plot_3d_hand(pts3d)

    return

# --- matplotlib リアルタイム初期化 ---
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# --- 3D描画関数 ---
def plot_3d_hand(pts3d):
    #plt.ion()
    #fig = plt.figure(figsize=(8, 6))
    #ax = fig.add_subplot(111, projection='3d')

    ax.cla()
    xs, ys, zs = pts3d[:, 0], pts3d[:, 1], pts3d[:, 2]

    ax.scatter(xs, ys, zs, c='b', marker='o')

    # ランドマーク番号を表示
    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
        ax.text(x, y, z, str(i), color='red', fontsize=8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Hand Landmarks")

    for start, end in HAND_CONNECTIONS:
        xline = [pts3d[start, 0], pts3d[end, 0]]
        yline = [pts3d[start, 1], pts3d[end, 1]]
        zline = [pts3d[start, 2], pts3d[end, 2]]
        ax.plot(xline, yline, zline, color='gray')

    ax.invert_xaxis()
    ax.invert_zaxis()

    plt.tight_layout()
    plt.pause(0.001)

while True:
    ret0, frame0 = cap0.read()
    ret2, frame2 = cap2.read()
    if not ret0 or not ret2:
        break

    img0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    
    hand_estimate(img0,img2)

    cv2.imshow('cam0', frame0)
    cv2.imshow('cam2', frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap0.release()
cap2.release()
cv2.destroyAllWindows()
plt.ioff()