import os 
import cv2
import numpy as np
from util import get_face_landmarks

data_dir = 'fer2013/train'
output = []
LIMIT_PER_EMOTION = 2000

# Define only the 3 classes you want
target_emotions = ['happy', 'sad', 'surprise']

for emotion_indx, emotion in enumerate(target_emotions):
    emotion_path = os.path.join(data_dir, emotion)
    
    # Safety check if folder exists
    if not os.path.exists(emotion_path):
        print(f"Skipping {emotion}, folder not found.")
        continue
        
    print(f"Processing: {emotion}...")
    count = 0 
    
    for image_path_ in os.listdir(emotion_path):
        if count >= LIMIT_PER_EMOTION:
            break 
            
        image_path = os.path.join(emotion_path, image_path_)
        image = cv2.imread(image_path)
        
        if image is None:
            continue
            
        face_landmarks = get_face_landmarks(image)
        
        if face_landmarks is not None and len(face_landmarks) == 1404:
            # util.py already does scale normalization - just use it directly
            normalized_list = face_landmarks.copy()
            
            # Append the label (0, 1, or 2)
            normalized_list.append(int(emotion_indx))
            output.append(normalized_list)
            count += 1

# Save the normalized 3-class data
np.savetxt('data.txt', np.array(output))
print(f"Done! Saved {len(output)} samples to data.txt")