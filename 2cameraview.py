import cv2

cap1 = cv2.VideoCapture(4)
cap2 = cv2.VideoCapture(6)

if not cap1.isOpened() or not cap2.isOpened():
    print("camera is not open")
    exit()

while True:
    ret1,frame1 = cap1.read()
    ret2,frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    cv2.imshow("Camera2",cv2.flip(frame2,-1))
    cv2.imshow("Camera1",cv2.flip(frame1,-1))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()