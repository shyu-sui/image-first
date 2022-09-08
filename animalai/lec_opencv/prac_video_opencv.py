import cv2

cap = cv2.VideoCapture('movie1.mp4')
# print(cap.isOpened())

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.resize(image, dsize=None, fx=0.03, fy=0.03)

    cv2.imshow('playshow', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
