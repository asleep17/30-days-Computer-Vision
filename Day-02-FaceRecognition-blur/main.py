import cv2
import mediapipe as mp
import argparse
import os
def process_image(img,face_detection):
     img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
     out =face_detection.process(img_rgb)
     H,W, _=img.shape
     if out.detections is not None:
        for detection in out.detections:
            location_data=detection.location_data
            bbox=location_data.relative_bounding_box
            x1,y1,w,h=bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1=int(x1*W)
            y1=int(y1*H)
            w=int(w*W)
            h=int(h*H)

            img[y1:y1+h, x1:x1+w,:] = cv2.blur(img[y1:y1+h, x1:x1+w,:], (30,30))
     return img

            #blur face


args=argparse.ArgumentParser()
args.add_argument("--mode",default='webcam')
args.add_argument("--filepath",default=None)

args= args.parse_args()

output_dir='./output'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#face detection
mp_face_detection=mp.solutions.face_detection

with mp_face_detection.FaceDetection(min_detection_confidence=0.5,model_selection=0) as face_detection:
    if args.mode in ['image']:

        img=cv2.imread(args.filepath)


        img=process_image(img,face_detection)
        cv2.imwrite( os.path.join(output_dir,'output.png'), img)
        
    elif args.mode in ['video']:
       
        cap=cv2.VideoCapture(args.filepath)
        ret,frame=cap.read()
        output_video=cv2.VideoWriter( os.path.join(output_dir,'output.mp4'),
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       25,
                                       (frame.shape[1],frame.shape[0]) )

        while ret:
            frame=process_image(frame,face_detection)
            output_video.write(frame)
            ret,frame=cap.read()

        cap.release()   
        output_video.release()
    elif args.mode in ['webcam']:
        cap=cv2.VideoCapture(0)

        while True:
            ret,frame=cap.read()
            if not ret:
                break
            frame=process_image(frame,face_detection)
            cv2.imshow('Webcam Face Blurring', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()