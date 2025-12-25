import mediapipe as mp
import pickle
import cv2
import os
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands= mp_hands.Hands(static_image_mode=True,
                      min_detection_confidence=0.3)

Data_dir='./data'
data=[]
label=[]
for dir_ in os.listdir(Data_dir):
    for img_path in os.listdir(os.path.join(Data_dir, dir_)):
        data_aux=[]
        img = cv2.imread(os.path.join(Data_dir, dir_, img_path))
        img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results=hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
               for i in range (len(hand_landmarks.landmark)):
                   x = hand_landmarks.landmark[i].x
                   y = hand_landmarks.landmark[i].y 
                   data_aux.append(x)
                   data_aux.append(y)

            data.append(data_aux)
            label.append(dir_)
       
f=open('data.pickle','wb')
pickle.dump({'data':data,'label':label},f)
f.close()
