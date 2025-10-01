import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import mediapipe as mp

# 座標データをファイルから読み込んで測定するやつ

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