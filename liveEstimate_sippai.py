import cv2
import pickle
import os
from datetime import datetime
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler

coorfilename0 = 'hand_coordinates0.pkl'
coorfilename2 = 'hand_coordinates2.pkl'
model_path = 'hand_landmarker.task'


BaseOptions   = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult  = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode     = mp.tasks.vision.RunningMode

global hands_xy0
global hands_xy2
hands_xy0 = []
hands_xy2 = []

# --- キャリブレーションマップ読み込み ---
with open('stereo_calibration_data.pkl', 'rb') as f:
    data = pickle.load(f)

P1 = data['P1']
P2 = data['P2']

# 画像サイズ
width, height = 640, 480

# =============== コールバック ==================
def print_result_factory(num):
    def print_result(result: HandLandmarkerResult, output_image: mp.Image, ts_ms: int):
        global X,Y,Z
        try:
            image_np = output_image.numpy_view().copy()
            height, width, _ = image_np.shape
            if num == 0:
                # 手の数だけ x, y を取り出す → (hand_count, 21, 2)
                for hand_landmarks in result.hand_landmarks:
                    one_hand = []
                    for lm in hand_landmarks:
                        x_px = int(lm.x * width)
                        y_px = int(lm.y * height)
                        one_hand.append([lm.x, lm.y])
                    hands_xy0.append(one_hand)

                # NumPy 配列化
                hands_xy0 = np.array(hands_xy0)  # shape: (N, 21, 2)
                print(f"hands_xy0 shape: {hands_xy0.shape}")
                print(hands_xy0)
            if num == 2:
                # 手の数だけ x, y を取り出す → (hand_count, 21, 2)
                for hand_landmarks in result.hand_landmarks:
                    one_hand = []
                    for lm in hand_landmarks:
                        x_px = int(lm.x * width)
                        y_px = int(lm.y * height)
                        one_hand.append([lm.x, lm.y])
                    hands_xy2.append(one_hand)

                # NumPy 配列化
                hands_xy2 = np.array(hands_xy2)  # shape: (N, 21, 2)
                print(f"hands_xy2 shape: {hands_xy2.shape}")
                print(hands_xy2)

        except Exception as e:
            print(f"Error in print_result: {e}")
        
    return print_result


callback0 = print_result_factory(num=0)
callback2 = print_result_factory(num=2)

# ============== HandLandmarker ================
options0 = HandLandmarkerOptions(
    base_options   = BaseOptions(model_asset_path=model_path),
    running_mode   = VisionRunningMode.LIVE_STREAM,
    result_callback = callback0,
    num_hands      = 1,
)

options2 = HandLandmarkerOptions(
    base_options   = BaseOptions(model_asset_path=model_path),
    running_mode   = VisionRunningMode.LIVE_STREAM,
    result_callback = callback2,
    num_hands      = 1,
)

# matplotlibセットアップ
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def draw(X,Y,Z):
    ax.scatter(X, Y, Z, c='blue', marker='o')  # 点群を描画

    # Mediapipeが提供する接続情報
    HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

    # 各接続ごとに線を引く
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection

        # 座標取得
        x_vals = [X[start_idx], X[end_idx]]
        y_vals = [Y[start_idx], Y[end_idx]]
        z_vals = [Z[start_idx], Z[end_idx]]

        # 線を描画
        ax.plot(x_vals, y_vals, z_vals, color='black', linewidth=1)

    # 軸ラベル
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 軸の向き調整（必要に応じて）
    ax.invert_yaxis()  # Mediapipe系データでは上下反転してることが多い

    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(0, 2)  # Zが奥行き

    plt.title('3D Hand Landmarks (Triangulated)')
    plt.show()


with HandLandmarker.create_from_options(options0) as landmarker0:

    cap0 = cv2.VideoCapture(0)
    if not cap0.isOpened():
        print('cannnot open camera0')

    with HandLandmarker.create_from_options(options2) as landmarker2:

        cap2 = cv2.VideoCapture(2)
        if not cap2.isOpened():
            print('cannnot open camera2')

        while True:
            ret,frame0 = cap0.read()
            frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
            mp_image0 = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame0)
            landmarker0.detect_async(mp_image0, int(time.time() * 1000))

            ret,frame2 = cap2.read()
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            mp_image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame2)
            landmarker2.detect_async(mp_image2, int(time.time() * 1000))

            cv2.imshow("Camera0",frame0)
            cv2.imshow("Camera2",frame2)

            #サイズ変更
            if hands_xy0 and hands_xy2:
                PP1 = np.reshape(hands_xy0,(21,2))
                PP2 = np.reshape(hands_xy2,(21,2))
                # 正規化座標 → ピクセル座標
                pts1_px = PP1 * np.array([width, height])
                pts2_px = PP2 * np.array([width, height])
                out = np.zeros((4,21),np.float32)

                out = cv2.triangulatePoints(
                    P1,
                    P2,
                    pts1_px.T,
                    pts2_px.T
                )

                points_3d = (out[:3, :] / out[3, :]).T  # (21, 3)

                print(points_3d)

                # points_3d.shape = (21, 3)
                scaler = MinMaxScaler()
                points_3d_scaled = scaler.fit_transform(points_3d)  # 0〜1 の範囲に正規化

                X, Y, Z = points_3d_scaled[:, 0], points_3d_scaled[:, 1], points_3d_scaled[:, 2]

                draw(X,Y,Z)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
cap0.release()
cap2.release()
cv2.destroyAllWindows()


