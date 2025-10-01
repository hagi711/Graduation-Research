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

# 画像サイズ（キャリブレーション時と同じ）
image_size = (640, 480)  # 例、適宜変更

camMtx1, dist1 = calib['cameraMatrix1'], calib['distCoeffs1']
camMtx2, dist2 = calib['cameraMatrix2'], calib['distCoeffs2']
R, T = calib['R'], calib['T']
P1, P2 = calib['P1'], calib['P2']
R1, R2 = calib['R1'], calib['R2']

# --- rectification maps ---
map1x, map1y = cv2.initUndistortRectifyMap(camMtx1, dist1, R1, P1, image_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(camMtx2, dist2, R2, P2, image_size, cv2.CV_32FC1)

# --- MediaPipe Hand ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # 動画対応
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands2 = mp_hands.Hands(
    static_image_mode=False,  # 動画対応
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# --- カメラ準備 ---
cap0 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

# 手ごとの3D座標を保存する辞書
landmarks_history = {
    'Left': None,   # 左手の最新3D座標
    'Right': None   # 右手の最新3D座標
}

# ファイル名変数
num = 0

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

def get_labeled_landmarks2(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands2.process(img_rgb)

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

def compute_joint_distances(pts3d, label=''):
    """
    pts3d: (21,3) 三角測量で得た手の3D座標
    HAND_CONNECTIONS: MediaPipeの関節接続リスト
    """
    distances = []
    for start, end in HAND_CONNECTIONS:
        d = np.linalg.norm(pts3d[start] - pts3d[end])
        distances.append(d)
    
    distances = np.array(distances)
    
    # 保存用グローバル辞書
    if 'joint_dist_stats' not in globals():
        global joint_dist_stats
        joint_dist_stats = {'Left': [], 'Right': []}
    
    joint_dist_stats[label].append(distances)
    
    # 結果を出力
    print(f"--- {label} hand ---")
    for i, d in enumerate(distances):
        print(f"Connection {i} distance: {d:.2f} mm")
    print(f"平均距離: {distances.mean():.2f} mm, 最大距離: {distances.max():.2f} mm\n")
    
    return distances

def hand_estimate(img0, img2):
    hands0 = get_labeled_landmarks(img0)
    hands2 = get_labeled_landmarks2(img2)
    
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

        # --- 三角測量 ---
        pts4d_hom = cv2.triangulatePoints(
            P1, P2,
            pts0.reshape(-1,2).T,
            pts2.reshape(-1,2).T)
        pts3d = (pts4d_hom[:3] / pts4d_hom[3]).T

        # 関節距離の安定性測定
        #compute_joint_distances(pts3d, label)

        #座標出力
        #for i, point in enumerate(pts3d):
        #   print(f"Point {i}: X={point[0]:.3f}, Y={point[1]:.3f}, Z={point[2]:.3f}")

        # 3Dプロット
        update_3d_hand(pts3d, label)

        # 最新の座標を保存
        landmarks_history[label] = pts3d
    
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
ax.set_xlim(-200, 0)
ax.set_ylim(-100, 100)
ax.set_zlim(-300, -500)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# --- 3D描画関数 ---
def update_3d_hand(pts3d, label=''):
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

    # --- 画像補正・整列 ---
    left  = cv2.remap(frame0,  map1x, map1y, cv2.INTER_LINEAR)
    right = cv2.remap(frame2, map2x, map2y, cv2.INTER_LINEAR)

    img0_dbg, img2_dbg = hand_estimate(cv2.flip(left,-1), cv2.flip(right,-1))
    #img0_dbg, img2_dbg = hand_estimate(left,right)

    cv2.imshow('cam0', img0_dbg)
    cv2.imshow('cam2', img2_dbg)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('o') or key == ord('s'):
        num = num + 1
        # 現在の座標を保存
        with open(f'coordinate{num}.pkl', 'wb') as f:
            pickle.dump(landmarks_history, f)
            print(landmarks_history)

        print(f"座標を保存しました: coordinate{num}.pkl")
    

cap0.release()
cap2.release()
cv2.destroyAllWindows()
plt.ioff()