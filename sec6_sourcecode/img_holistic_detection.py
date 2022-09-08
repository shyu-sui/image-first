import mediapipe as mp
import cv2

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=2, color=(0, 255, 0))
mark_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0, 0, 255))

img_path = 'holistic.jpg'

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        static_image_mode=True) as holistic_detection:
    image = cv2.imread(img_path)
    image = cv2.resize(image, dsize=None, fx=0.3, fy=0.3)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height = rgb_image.shape[0]
    width = rgb_image.shape[1]

    results = holistic_detection.process(rgb_image)

    annotated_image = image.copy()

    print(results.face_landmarks.landmark[0].x)
    print(results.pose_landmarks.landmark[11].x)#left shoulder
    print(results.right_hand_landmarks.landmark[0].x)#wrist

    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=results.face_landmarks,
        connections=mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=mark_drawing_spec,
        connection_drawing_spec=mesh_drawing_spec
        )
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=results.pose_landmarks,
        connections=mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mark_drawing_spec,
        connection_drawing_spec=mesh_drawing_spec
        )
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=results.left_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mark_drawing_spec,
        connection_drawing_spec=mesh_drawing_spec
        )
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=results.right_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mark_drawing_spec,
        connection_drawing_spec=mesh_drawing_spec
        )
    cv2.imwrite('result.jpg', annotated_image)
