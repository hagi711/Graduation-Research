import numpy as np
import cv2 as cv
import glob
import pickle
import sys

# Chessboard settings
chessboardSize = (6, 5)

args = sys.argv
port1 = int(args[1])
port2 = int(args[2])
framesize = np.array([640,480])

# Termination criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (3D points in real world space)
objp1 = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp1[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

# Size of chessboard squares in mm
size_of_chessboard_squares_mm = 23
objp1 = objp1 * size_of_chessboard_squares_mm
objp2 = objp1

# Arrays to store object points and image points
objpoints1 = []  # 3D points in real world space
imgpoints1 = []  # 2D points in image plane
objpoints2 = []  # 3D points in real world space
imgpoints2 = []  # 2D points in image plane

# Load chessboard images
images1 = glob.glob(f'./images/camera{port1}/*.png')
images2 = glob.glob(f'./images/camera{port2}/*.png')

for image in images1:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If corners are found, refine them and store the points
    if ret:
        objpoints1.append(objp1)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints1.append(corners)

        # Draw and display the corners
        #cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        #cv.imshow('img', img)
        #cv.waitKey(500)

for image in images2:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If corners are found, refine them and store the points
    if ret:
        objpoints2.append(objp2)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints2.append(corners)

        # Draw and display the corners
        #cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        #cv.imshow('img', img)
        #cv.waitKey(500)

cv.destroyAllWindows()

# Camera calibration
ret, cameraMatrix1, dist1, rvecs1, tvecs1 = cv.calibrateCamera(objpoints1, imgpoints1, framesize, None, None)
ret, cameraMatrix2, dist2, rvecs2, tvecs2 = cv.calibrateCamera(objpoints2, imgpoints2, framesize, None, None)

# 保存ファイル名
filename = 'calibration_data_debug.pkl'

# 保存する辞書
data = {
    'cameraMatrix1':cameraMatrix1,
    'cameraMatrix2':cameraMatrix2,
    'distCoeffs1':dist1,
    'distCoeffs2':dist2
}

# ファイルに保存
with open(filename, 'wb') as f:
    pickle.dump(data, f)

print(f"Camera matrix{port1} : \n")
print(cameraMatrix1)
print(f"dist{port1} : \n")
print(dist1)

print(f"Camera matrix{port2} : \n")
print(cameraMatrix2)
print(f"dist{port1} : \n")
print(dist2)

print("Camera calibration completed and saved.")