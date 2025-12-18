import cv2
import numpy as np
import mediapipe as mp
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from datetime import datetime
from collections import deque

np.set_printoptions(precision=3,suppress=True)

ports = [0, 2]
caps = []
nums = {0: 0, 2: 0}

# --- カメラ準備 ---
cap0 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

# --- キャリブレーション読み込み ---
#with open('stereo_calibration_data.pkl', 'rb') as f:
#    calib = pickle.load(f)

with open('stereo_calibration_data_circle.pkl', 'rb') as f:
    calib = pickle.load(f)

camMtx1, dist1 = calib['cameraMatrix1'], calib['distCoeffs1']
camMtx2, dist2 = calib['cameraMatrix2'], calib['distCoeffs2']
R, T = calib['R'], calib['T']
R1, R2 = calib['R1'], calib['R2']

# --- キャリブレーション読み込み(世界座標変換用) ---
with open('chessboard.pkl', 'rb') as f:
    chess = pickle.load(f)

objectpoints = chess['objpts']
imgpoints = chess['imgpts']
objpts = np.array(objectpoints, dtype=np.float32).reshape(-1,3)
imgpts = np.array(imgpoints, dtype=np.float32).reshape(-1,2)

# --- MediaPipe Hand ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # 動画対応
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 手ごとの3D座標を保存する辞書
landmarks_history = {
    'Left': None,   # 左手の最新3D座標
    'Right': None   # 右手の最新3D座標
}

#10フレーム分の座標を保存
landmarks_buffer = {
    "Left": deque(maxlen=10),
    "Right": deque(maxlen=10)
}

# グローバル変数として前回の平均を保持
prev_mean_landmarks = {'Left': None, 'Right': None}

# --- 状態を保持するフラグ ---
mouse_state = {'save': False, 'quit': False}

# --- マウスイベントコールバック関数 ---
def mouse_callback(event, x, y, flags, param):
    global mouse_state
    global previous_distances
    if event == cv2.EVENT_LBUTTONDOWN:  # 左クリック
        mouse_state['save'] = True
    elif event == cv2.EVENT_MBUTTONDOWN:  # 中央クリック
        mouse_state['quit'] = True


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

hands2 = mp_hands.Hands(
    static_image_mode=False,  # 動画対応
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

# 距離変化保存用
joint_distance_changes = {'Left': [], 'Right': []}
previous_distances = {'Left': None, 'Right': None}

def compute_joint_distance_changes(pts3d, label=''):
    """
    pts3d: (21,3) 三角測量で得た手の3D座標
    label: 'Left' or 'Right'
    """
    global previous_distances, joint_distance_changes

    # 現在のフレームの距離（いつもの計算）
    current_distances = []
    for start, end in HAND_CONNECTIONS:
        d = np.linalg.norm(pts3d[start] - pts3d[end])
        current_distances.append(d)
    current_distances = np.array(current_distances)

    # 初回フレーム：
    if previous_distances[label] is None:
        previous_distances[label] = current_distances
        print(f"[{label}] 初回フレーム: 距離変化 = 0 (基準フレーム保存)")
        joint_distance_changes[label].append(np.zeros_like(current_distances))
        return current_distances

    # 距離変化（Δ距離 = 今 - 前フレーム）
    distance_change = current_distances - previous_distances[label]
    joint_distance_changes[label].append(distance_change)

    # 前フレーム更新
    previous_distances[label] = current_distances

    # --- 出力 ---
    '''
    if max(distance_change) > 2.0:
        print(f"--- {label} hand 距離変化 Δ ---")
        for i, d in enumerate(distance_change):
            print(f"Connection {i}: Δdistance = {d:.2f} mm")
        print(f"平均変化: {distance_change.mean():.2f} mm, 最大変化: {np.max(np.abs(distance_change)):.2f} mm\n")
    '''

    return current_distances

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

        # 歪み補正
        pts0_norm = cv2.undistortPoints(pts0.astype(np.float64), camMtx1, dist1, P=None)
        pts2_norm = cv2.undistortPoints(pts2.astype(np.float64), camMtx2, dist2, P=None)
 

        #射影行列
        P1 = np.hstack((np.eye(3), np.zeros((3,1))))
        P2 = np.hstack((R, T.reshape(3,1)))

        # --- 三角測量 ---
        pts4d_hom = cv2.triangulatePoints(
            P1, P2,
            pts0_norm.reshape(-1,2).T,
            pts2_norm.reshape(-1,2).T)
        pts3d = (pts4d_hom[:3] / pts4d_hom[3])

        # --- 世界座標変換 ---
        retval, rvec_left, tvec_left = cv2.solvePnP(objpts, imgpts,
                                           camMtx1, dist1)
        R_left, _ = cv2.Rodrigues(rvec_left)  # (3,3)
        t_left = tvec_left.reshape(3,1)      # (3,1)

        # --- カメラ座標 -> ワールド座標 ---
        # X_world = R^T (X_cam - t)
        pts3d_world = R_left.T @ (pts3d - t_left)  # (3, N)
        
        # 関節距離の安定性測定
        compute_joint_distance_changes(pts3d_world.T, label)

        #pts3d表示
        #for i, point in enumerate(pts3d_world.T):
        #   print(f"Point {i}: X={point[0]:.3f}, Y={point[1]:.3f}, Z={point[2]:.3f}")

        # 最新の座標を保存
        landmarks_history[label] = pts3d_world.T

        # 現在のランドマークをバッファに追加
        for hand in ['Left', 'Right']:
            if landmarks_history[hand] is not None:
                landmarks_buffer[hand].append(landmarks_history[hand])

        # 3Dプロット
        #update_3d_hand(pts3d_world.T, label)
    
    return img0_dbg,img2_dbg


def analyze_joint_distance_changes():
    """
    フレームごとに記録された joint_dist_stats から
    各接続ごとの距離変化を解析・可視化する
    """
    
    for label, all_dists in joint_distance_changes.items():
        if len(all_dists) < 2:
            print(f"{label} hand: フレーム数が少なすぎます。")
            continue

        # 各フレームの距離変化量を計算
        all_dists = np.array(all_dists)  # shape = (num_frames, num_connections)
        diff = np.abs(np.diff(all_dists, axis=0))  # 隣接フレーム間の変化量

        # 平均変化量が大きい接続を特定
        mean_changes = diff.mean(axis=0)
        top_indices = np.argsort(mean_changes)[::-1][:20]
        top_connections = np.array(list(HAND_CONNECTIONS))[top_indices]

        print(f"\n=== {label} hand ===")
        for (i, (s, e)) in enumerate(top_connections):
            print(f"接続 {i+1}: connection {s}-{e}, 平均変化量 = {mean_changes[top_indices[i]]:.3f} mm")

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

#軸範囲を固定
ax.set_xlim(-100,100)
ax.set_ylim(-200,0)
ax.set_zlim(-100,100)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# --- 3D描画関数 ---
def update_3d_hand(pts3d, label=''):
    # Y,Z軸を反転（自分から見た方向に合わせる）
    pts3d[:,1] = -pts3d[:,1]
    pts3d[:,2] = -pts3d[:,2]
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
    
    img0_dbg, img2_dbg = hand_estimate(frame0,frame2)
    cv2.imshow('cam0', img0_dbg)
    cv2.imshow('cam2', img2_dbg)

    cv2.setMouseCallback("cam0", mouse_callback)
    cv2.setMouseCallback("cam2", mouse_callback)

    frames = {}
    frames[0] = img0_dbg
    frames[2] = img2_dbg

    k = cv2.waitKey(5) & 0xFF
    if mouse_state['quit']:
        break
    elif mouse_state['save']:
        # 保存処理
        mouse_state['save'] = False
        # バッファが満たされたら平均を計算
        for hand in ['Left']:
            if len(landmarks_buffer[hand]) == 10:
                stacked = np.stack(landmarks_buffer[hand], axis=0)  # (frame, 21, 3)
                mean_landmarks = np.mean(stacked, axis=0)           # (21, 3)

                if prev_mean_landmarks[hand] is None:
                    print("previous landmarks saved")
                else:
                    diff = mean_landmarks - prev_mean_landmarks[hand]
                    #print(f"\n=== {hand} hand ===                      === 変化量 ===")
                    for i, (coord, dist) in enumerate(zip(mean_landmarks, diff)):
                        if i == 8:
                            print(f'landmark {i}: {coord}  change {i} : {dist}')
                        '''
                    for i, coord in enumerate(mean_landmarks):
                        print(f'landmark {i}: {coord}')
                    
                    for i,dist in enumerate(diff):
                        print(f"change dist {i} : {dist} ")
                        '''
                
            # 現在の平均を保存（次回比較用）
            prev_mean_landmarks[hand] = mean_landmarks.copy()

        '''
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for port in ports:
                if port in frames:
                    filename = f'./images/camera{port}/img{nums[port]}_{timestamp}.png'
                    cv2.imwrite(filename, frames[port])
                    print(f"Camera {port} - 保存: {filename}")
                    nums[port] += 1
        '''

cap0.release()
cap2.release()
cv2.destroyAllWindows()
plt.ioff()

# グラフ描画
#plot_joint_distance_changes()

#各関節の平均変化
#analyze_joint_distance_changes()