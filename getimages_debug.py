import cv2
import pickle
import os
from datetime import datetime

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

def main():
    ports = [4, 6]
    caps = []
    nums = {4: 0, 6: 0}

    # Open both cameras
    for port in ports:
        cap = cv2.VideoCapture(port)
        if not cap.isOpened():
            print(f"カメラ{port}が開けません")
            return
        save_frame_size(cap, port)
        caps.append((port, cap))
        os.makedirs(f'./images/camera{port}', exist_ok=True)

    print("s: 両カメラの画像を保存, q: 終了")

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
        if k == ord('q') or k == 27:
            break
        elif k == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for port in ports:
                if port in frames:
                    filename = f'./images/camera{port}/img{nums[port]}_{timestamp}.png'
                    cv2.imwrite(filename, frames[port])
                    print(f"Camera {port} - 保存: {filename}")
                    nums[port] += 1

    for _, cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
