import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.resize(image, dsize=None, fx=0.5, fy=0.5)

    imgae = cv2.flip(image, -1)

    cv2.imshow('playshow', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
