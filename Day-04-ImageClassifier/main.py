import os
import numpy as np
from numpy import resize

from skimage.io import imread
from skimage.transform import resize

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

input_dir='C:/Users/User/30-days-Computer-Vision/Day-04-ImageClassifier/clf-data'
categories=['empty','not_empty']
data=[]
labels=[]
for categoryindx,category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir,category)):
        img_path=os.path.join(input_dir,category,file)
        img=imread(img_path)
        img= resize(img,(15,15))
        data.append(img.flatten())
        labels.append(categoryindx)

data=np.array(data)
labels=np.array(labels)

x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.2,shuffle=True,stratify=labels,random_state=42)

classifier=SVC()

parameters=[{'gamma':[0.01,0.001,0.0001],'C':[1,10,100,1000]}]

grid_searchCV=GridSearchCV(classifier,parameters)

grid_searchCV.fit(x_train,y_train)

best_estimator=grid_searchCV.best_estimator_
y_pred=best_estimator.predict(x_test)

score=accuracy_score(y_pred,y_test)
import pickle

# Save the model to a file
with open('model.p', 'wb') as f:
    pickle.dump(best_estimator, f)

print("Model saved as model.p")
from sklearn.metrics import classification_report

# This prints the beautiful table with Precision, Recall, and F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=categories))