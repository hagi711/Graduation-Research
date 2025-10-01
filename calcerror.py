import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- ファイルパスを指定 ---
file1 = "coordinate1.pkl"
file2 = "coordinate2.pkl"

# --- pickle読み込み ---
def load_hand_landmarks(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

landmarks1 = load_hand_landmarks(file1)
landmarks2 = load_hand_landmarks(file2)

for hand in ['Left', 'Right']:
    pts1 = landmarks1.get(hand)
    pts2 = landmarks2.get(hand)
    
    if pts1 is None or pts2 is None:
        print(f"{hand} hand data missing in one of the files.")
        continue
    
    delta = pts2 - pts1
    dx, dy, dz = delta[:,0], delta[:,1], delta[:,2]
    distances = np.linalg.norm(delta, axis=1)

    print(f"\n--- {hand} Hand Landmark Movements ---")
    print("Index | ΔX      ΔY      ΔZ      3D distance")
    for i, (x, y, z, d) in enumerate(zip(dx, dy, dz, distances)):
        print(f"{i:2d}    {x:7.3f} {y:7.3f} {z:7.3f} {d:10.3f}")

    # --- 棒グラフ描画 ---
    indices = np.arange(len(pts1))  # ランドマーク番号
    width = 0.2
    plt.figure(figsize=(12,5))
    plt.bar(indices - 1.5*width, dx, width, label='ΔX')
    plt.bar(indices - 0.5*width, dy, width, label='ΔY')
    plt.bar(indices + 0.5*width, dz, width, label='ΔZ')
    plt.bar(indices + 1.5*width, distances, width, label='3D distance', color='gray', alpha=0.5)
    
    plt.xticks(indices, [str(i) for i in indices])
    plt.xlabel("Landmark Index")
    plt.ylabel("Movement (mm)")
    plt.title(f"{hand} Hand Landmark Movement")
    plt.legend()
    plt.grid(axis='y')
    plt.show()