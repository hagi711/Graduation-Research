import pickle
import numpy as np

# pklファイル読み込み
with open("stereo_calibration_data.pkl", "rb") as f:
    data = pickle.load(f)

# 全てのデータをテキストで表示
for key, value in data.items():
    print(f"--- {key} ---")
    if isinstance(value, np.ndarray):
        # 配列は整形して表示
        print(np.array2string(value, precision=4, suppress_small=True))
    else:
        # その他の型はそのまま表示
        print(value)