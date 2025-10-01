import cv2
import numpy as np
import mediapipe as mp
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS


# --- キャリブレーション読み込み ---
with open('stereo_calibration_data_debug.pkl', 'rb') as f:
    calib = pickle.load(f)

camMtx1, dist1 = calib['cameraMatrix1'], calib['distCoeffs1']
camMtx2, dist2 = calib['cameraMatrix2'], calib['distCoeffs2']
R, T = calib['R'], calib['T']
R1, R2 = calib['R1'], calib['R2']

# --- MediaPipe Hand ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # 動画対応
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- カメラ準備 ---
cap0 = cv2.VideoCapture(4)
cap2 = cv2.VideoCapture(6)

def get_labeled_landmarks(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    landmarks_list = []
    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label  # 'Left' / 'Right'
            score = handedness.classification[0].score
            #ラベルを自分から見た左右に反転
            label = 'Right' if label == 'Left' else 'Left'
            landmarks = [(lm.x * img.shape[1], lm.y * img.shape[0]) for lm in hand_landmarks.landmark]
            landmarks_list.append({
                "label": label,
                "score": score,
                "landmarks": np.array(landmarks, dtype=np.float32)
            })
            
    return landmarks_list

def draw_landmarks_debug(img, landmarks_list):
    """
    MediaPipeの出力をデバッグ用に画像へ描画
    """
    for h in landmarks_list:
        label = h["label"]
        pts = h["landmarks"]
        color = (0,0,255) if label == "Right" else (255,0,0)  # 赤=右手, 青=左手

        # 各ランドマークを描画
        for (x,y) in pts.astype(int):
            cv2.circle(img, (x,y), 3, color, -1)

        # 手のラベルを大きく表示
        x0, y0 = pts[0].astype(int)
        cv2.putText(img, label, (x0-20,y0-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return img

def hand_estimate(img0, img2):
    hands0 = get_labeled_landmarks(img0)
    hands2 = get_labeled_landmarks(img2)

     # --- デバッグ描画をフレームに追加 ---
    img0_dbg = draw_landmarks_debug(img0.copy(), hands0)
    img2_dbg = draw_landmarks_debug(img2.copy(), hands2)

    for label in ['Left','Right']:
        h0 = next((h for h in hands0 if h['label'] == label), None)
        h2 = next((h for h in hands2 if h['label'] == label), None)

        if h0 is None or h2 is None:
            continue  # 両方揃わなければスキップ

        pts0 = h0['landmarks'].reshape(-1, 1, 2)
        pts2 = h2['landmarks'].reshape(-1, 1, 2)

        # 歪み補正
        pts0_norm = cv2.undistortPoints(pts0.reshape(-1,1,2).astype(np.float64),
                                    camMtx1, dist1, P=None).reshape(-1,2)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1,1,2).astype(np.float64),
                                    camMtx2, dist2, P=None).reshape(-1,2)

        # pts0_norm, pts2_norm をピクセル座標に変換して描画
        pts0_pix = cv2.convertPointsToHomogeneous(pts0_norm)[:,0,:]   # (N,3)
        pts0_pix = (camMtx1 @ pts0_pix.T).T                           # (N,3)
        pts0_pix = (pts0_pix[:,:2] / pts0_pix[:,2,np.newaxis]).astype(int)

        pts2_pix = cv2.convertPointsToHomogeneous(pts2_norm)[:,0,:]
        pts2_pix = (camMtx2 @ pts2_pix.T).T
        pts2_pix = (pts2_pix[:,:2] / pts2_pix[:,2,np.newaxis]).astype(int)

        # カメラ画面に描画
        for (x,y) in pts0_pix:
            cv2.circle(img0_dbg, (x,y), 3, (0,255,0), -1)
        for (x,y) in pts2_pix:
            cv2.circle(img2_dbg, (x,y), 3, (0,255,0), -1)

        # --- 射影行列---
        P1 = np.hstack((np.eye(3), np.zeros((3,1)))) # [I|0]
        P2 = np.hstack((R, T.reshape(3,1)))            # [R|t]

        # --- 三角測量 ---
        pts4d_hom = cv2.triangulatePoints(P1, P2, pts0_norm.T.astype(np.float64), pts2_norm.T.astype(np.float64))
        pts3d = (pts4d_hom[:3] / pts4d_hom[3]).T# pts0_norm, pts2_norm をピクセル座標に変換して描画
        
        pts0_pix = cv2.convertPointsToHomogeneous(pts0_norm)[:,0,:]   # (N,3)
        pts0_pix = (camMtx1 @ pts0_pix.T).T                           # (N,3)
        pts0_pix = (pts0_pix[:,:2] / pts0_pix[:,2,np.newaxis]).astype(int)

        pts2_pix = cv2.convertPointsToHomogeneous(pts2_norm)[:,0,:]
        pts2_pix = (camMtx2 @ pts2_pix.T).T
        pts2_pix = (pts2_pix[:,:2] / pts2_pix[:,2,np.newaxis]).astype(int)

        # カメラ画面に描画
        for (x,y) in pts0_pix:
            cv2.circle(img0_dbg, (x,y), 3, (0,255,0), -1)
        for (x,y) in pts2_pix:
            cv2.circle(img2_dbg, (x,y), 3, (0,255,0), -1)
        print(pts3d)

        #for i, point in enumerate(pts3d):
        #   print(f"Point {i}: X={point[0]:.3f}, Y={point[1]:.3f}, Z={point[2]:.3f}")

        # 3Dプロット
        update_3d_hand(pts3d, label)
    
    return img0_dbg,img2_dbg

# --- matplotlib リアルタイム初期化 ---
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1]) 

# 事前に scatter と line を作成
scatter_left = ax.scatter([], [], [], c='blue', marker='o', label="Left")
lines_left = [ax.plot([0,0],[0,0],[0,0], color='blue')[0] for _ in HAND_CONNECTIONS]

scatter_right = ax.scatter([], [], [], c='red', marker='o', label="Right")
lines_right = [ax.plot([0,0],[0,0],[0,0], color='red')[0] for _ in HAND_CONNECTIONS]

ax.set_title("3D Hand Landmarks (Both Hands)")
ax.legend()

# --- 軸範囲を固定（必要に応じて調整）
ax.set_xlim(-250, 250)
ax.set_ylim(-250, 250)
ax.set_zlim(0,-500)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# --- 3D描画関数 ---
def update_3d_hand(pts3d, label=''):
    # Y軸を反転（自分から見た左右に合わせる）
    pts3d[:,1] = -pts3d[:,1]
    X, Y, Z = pts3d[:, 0], pts3d[:, 1], pts3d[:, 2]

    if label == 'Left':
        scatter_left._offsets3d = (X, Y, Z)
        for (line, (start, end)) in zip(lines_left, HAND_CONNECTIONS):
            line.set_data([X[start], X[end]], [Y[start], Y[end]])
            line.set_3d_properties([Z[start], Z[end]])
    elif label == 'Right':
        scatter_right._offsets3d = (X, Y, Z)
        for (line, (start, end)) in zip(lines_right, HAND_CONNECTIONS):
            line.set_data([X[start], X[end]], [Y[start], Y[end]])
            line.set_3d_properties([Z[start], Z[end]])

    plt.pause(0.001)

while True:
    ret0, frame0 = cap0.read()
    ret2, frame2 = cap2.read()

    if not ret0 or not ret2:
        break
    
    img0_dbg, img2_dbg = hand_estimate(cv2.flip(frame0,-1), cv2.flip(frame2,-1))

    cv2.imshow('cam0', img0_dbg)
    cv2.imshow('cam2', img2_dbg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap0.release()
cap2.release()
cv2.destroyAllWindows()
plt.ioff()