import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score ,classification_report
data_dict=pickle.load(open('data.pickle','rb'))

data=np.asarray(data_dict['data'])
label=np.asarray(data_dict['label'])

X_train, X_test, y_train, y_test = train_test_split(data, label,test_size=0.2, shuffle=True,stratify=label)

model=RandomForestClassifier()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

f=open('hand_classifier.pkl','wb')
pickle.dump(model,f)
f.close()