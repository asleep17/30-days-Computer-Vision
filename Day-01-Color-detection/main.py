import cv2
from PIL import Image
from util import get_limits
cap=cv2.VideoCapture(0)

blue=[255,0,0]
while(True):
    ret,frame=cap.read()
    hsvimage=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_limit,upper_limit=get_limits(blue)
    mask=cv2.inRange(hsvimage,lower_limit,upper_limit)
    mask_=Image.fromarray(mask)
    bboxes=mask_.getbbox()
    if bboxes is not None:
        x1,y1,x2,y2=bboxes
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),5) 

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
