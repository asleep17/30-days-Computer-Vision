import pickle
import cv2
import numpy as np
from util import get_face_landmarks
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Labels
emotions = ['Happy', 'Sad', 'Surprise']

# Load model
with open('emotion_model.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_landmarks = get_face_landmarks(frame, draw=True)

    if face_landmarks and len(face_landmarks) == 1404:
        # util.py already does scale normalization - use directly
        processed_landmarks = face_landmarks
        
        # Predict
        output = model.predict([processed_landmarks])
        probs = model.predict_proba([processed_landmarks])
        
        label = emotions[int(output[0])]
        confidence = np.max(probs)

        # Print debug info
        print(f"Pred: {label} | Conf: {confidence:.2f}")

        # Draw the result
        color = (0, 255, 0) if label == 'Happy' else (255, 255, 0)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
    else:
        cv2.putText(frame, "No Face", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()