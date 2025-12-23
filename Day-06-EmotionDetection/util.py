import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh( 
    max_num_faces=1,
    min_detection_confidence=0.5
)

def get_face_landmarks(image, draw=False, Static_image_mode=True):
    # Handle Grayscale and Upscale
    image_resized = cv2.resize(image, (200, 200), interpolation=cv2.INTER_CUBIC)
    
    if len(image_resized.shape) == 2:
        image_input_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
    else:
        image_input_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Process
    results = face_mesh.process(image_input_rgb)
    image_landmarks = []

    if results.multi_face_landmarks:
        face_lms = results.multi_face_landmarks[0].landmark
        
        # Extract all coordinates
        coords = np.array([[lm.x, lm.y, lm.z] for lm in face_lms])
        
        # 1. Center: subtract mean (moves center to origin)
        coords = coords - coords.mean(axis=0)
        
        # 2. Scale normalization: divide by standard deviation
        # This makes the face size consistent regardless of distance
        std = coords.std()
        if std > 0:
            coords = coords / std
        
        # Flatten to list
        image_landmarks = coords.flatten().tolist()

        if draw:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                image=image_input_rgb,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_CONTOURS)
            cv2.imshow("Debug Landmarks", image_input_rgb)

    return image_landmarks