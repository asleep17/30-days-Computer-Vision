import cv2
import easyocr
import matplotlib.pyplot as plt

image_path = './data/img792.jpg'
img = cv2.imread(image_path)
reader = easyocr.Reader(['en'], gpu=False)

# Use a distinct name for the list of results
results = reader.readtext(img)

for res in results:
    bbox, detected_text, conf = res

    p1 = (int(bbox[0][0]), int(bbox[0][1])) # Top Left
    p2 = (int(bbox[2][0]), int(bbox[2][1])) # Bottom Right
    
    # Draw the rectangle
    if conf > 0.25:
     cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
     cv2.putText(img, detected_text, (p1[0], p1[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Convert BGR to RGB for matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()