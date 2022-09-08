from typing import Annotated
import mediapipe as mp
import cv2

mp_face_detect = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

img_path = 'miakarifa_6.jpg'

with mp_face_detect.FaceDetection(min_detection_confidence=0.9) as face_detection:
  image = cv2.imread(img_path)
  image = cv2.resize(image, dsize=None, fx=0.3, fy = 0.3)
  rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  height = rgb_image.shape[0]
  width = rgb_image.shape[1]

  results = face_detection.process(rgb_image)

  annotated_image = image.copy()

  for detection in results.detections:
    print(mp_face_detect.get_key_point(detection,mp_face_detect.FaceKeyPoint.NOSE_TIP))
    nose_x = mp_face_detect.get_key_point(detection,mp_face_detect.FaceKeyPoint.NOSE_TIP).x
    nose_y = mp_face_detect.get_key_point(detection,mp_face_detect.FaceKeyPoint.NOSE_TIP).y

    re_nose_x = int(nose_x * width)
    re_nose_y = int(nose_y * height)
    print(re_nose_x, re_nose_y)

    mp_drawing.draw_detection(annotated_image, detection)
    
    cv2.imwrite('miakarif.jpg', annotated_image)


