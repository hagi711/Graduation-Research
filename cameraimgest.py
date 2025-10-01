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

# カメラからsキーで写真を撮って測定するコード

global imgfilename0
global imgfilename2

def save_frame_size(cap, port):
    # Get frame size
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    print(f"Camera {port} frame size: {frame_size}")

    # Save frame size to file
    os.makedirs('./calib_files', exist_ok=True)
    with open(f'./calib_files/frameSize{port}.pkl', 'wb') as f:
        pickle.dump(frame_size, f)


ports = [0, 2]
caps = []
nums = {0: 0, 2: 0}

# Open both cameras
for port in ports:
    cap = cv2.VideoCapture(port)
    save_frame_size(cap, port)
    caps.append((port, cap))
    os.makedirs(f'./images/camera{port}', exist_ok=True)

print("s: 両カメラの画像を保存")

while True:
    frames = {}
    for port, cap in caps:
        ret, frame = cap.read()
        if not ret:
            print(f"カメラ{port}からフレーム取得失敗")
            continue
        frames[port] = frame
        cv2.imshow(f"Camera {port}", frame)

    k = cv2.waitKey(5) & 0xFF
    if k == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for port in ports:
            if port in frames:
                if port == 0:
                    imgfilename0 = f'./images/camera{port}/img{nums[port]}_{timestamp}.png'
                    cv2.imwrite(imgfilename0, frames[port])
                    print(f"Camera {port} - 保存: {imgfilename0}")
                if port == 2:
                    imgfilename2 = f'./images/camera{port}/img{nums[port]}_{timestamp}.png'
                    cv2.imwrite(imgfilename2, frames[port])
                    print(imgfilename2)
                nums[port] += 1
        break

for _, cap in caps:
    cap.release()
cv2.destroyAllWindows()




coorfilename0 = 'hand_coordinates0.pkl'
coorfilename2 = 'hand_coordinates2.pkl'

model_path = 'hand_landmarker.task'


BaseOptions   = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult  = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode     = mp.tasks.vision.RunningMode
        
annotated_image = None

# =============== コールバック ==================
def print_result_factory(num):
    def print_result(result: HandLandmarkerResult, output_image: mp.Image, ts_ms: int):
        global annotated_image

        try:
            image_np = output_image.numpy_view().copy()
            height, width, _ = image_np.shape
            # 手の数だけ x, y を取り出す → (hand_count, 21, 2)
            hands_xy = []

            for hand_landmarks in result.hand_landmarks:
                one_hand = []
                for lm in hand_landmarks:
                    x_px = int(lm.x * width)
                    y_px = int(lm.y * height)

                    # 画像に円を描く
                    cv2.circle(image_np, (x_px, y_px), radius=3, color=(0, 255, 0), thickness=-1)

                    one_hand.append([lm.x, lm.y])
                hands_xy.append(one_hand)

            # NumPy 配列化
            hands_xy = np.array(hands_xy)  # shape: (N, 21, 2)
            print(f"hands_xy shape: {hands_xy.shape}")
            print(hands_xy)

            # 保存する辞書
            data = {
            'hand_coordinate' : hands_xy
            }

            # ファイルに保存
            
            if num == 0:
                with open(coorfilename0, 'wb') as f:
                    pickle.dump(data, f)
            if num == 2:
                with open(coorfilename2, 'wb') as f:
                    pickle.dump(data, f)
            

            annotated_image = image_np

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
    num_hands      = 2,
)

options2 = HandLandmarkerOptions(
    base_options   = BaseOptions(model_asset_path=model_path),
    running_mode   = VisionRunningMode.LIVE_STREAM,
    result_callback = callback2,
    num_hands      = 2,
)


with HandLandmarker.create_from_options(options0) as landmarker:

    frame0 = cv2.imread(imgfilename0)
    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)

    mp_image0 = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame0)
    landmarker.detect_async(mp_image0, int(time.time() * 1000))

with HandLandmarker.create_from_options(options2) as landmarker:

    frame2 = cv2.imread(imgfilename2)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    mp_image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame2)
    landmarker.detect_async(mp_image2, int(time.time() * 1000))

# --- キャリブレーションマップ読み込み ---
with open('stereo_calibration_data.pkl', 'rb') as f:
    data = pickle.load(f)

# ---- 座標読み込み --- 
with open('hand_coordinates0.pkl' , 'rb') as f:
  data2 = pickle.load(f)

with open('hand_coordinates2.pkl' , 'rb') as f:
  data3 = pickle.load(f)

P1 = data['P1']
P2 = data['P2']
coor0 = data2['hand_coordinate']
coor2 = data3['hand_coordinate']


PP1 = np.reshape(coor0,(21,2))
PP2 = np.reshape(coor2,(21,2))
# 画像サイズ
width, height = 640, 480

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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

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
