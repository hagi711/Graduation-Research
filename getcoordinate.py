import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import time
import csv
from mediapipe.framework.formats import landmark_pb2
import pickle
import sys


args = sys.argv
num = int(args[1])

filename = (f'hand_coordinates{num}.pkl')

model_path = 'hand_landmarker.task'


BaseOptions   = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult  = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode     = mp.tasks.vision.RunningMode

# ================= CSV 初期化 =================
header = []
for h in range(2):          # hand 0,1
    for i in range(21):     # landmark 0–20
        header += [f'h{h}_x{i}', f'h{h}_y{i}', f'h{h}_z{i}']

csv_file   = open('hand_landmarks.csv', 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(header)

annotated_image = None

# =============== コールバック ==================
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
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

        # 必要に応じて保存
        # 1フレーム = 1行に flatten して書き出し
        csv_writer.writerow(hands_xy.flatten())

        annotated_image = image_np

    except Exception as e:
        print(f"Error in print_result: {e}")

# ============== HandLandmarker ================
options = HandLandmarkerOptions(
    base_options   = BaseOptions(model_asset_path=model_path),
    running_mode   = VisionRunningMode.LIVE_STREAM,
    result_callback = print_result,
    num_hands      = 2,
)

# ============== 実行ループ =====================
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        '''
        ret, frame = cap.read()
        if not ret:
            break
        '''
        
        frame = cv2.imread(f'./images/camera{num}/img0_20250723_135623.png')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        while True:
            if annotated_image is not None:
                cv2.imshow("MediaPipe Hands", annotated_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            else:
                continue
        break



cap.release()
cv2.destroyAllWindows()
csv_file.close()
