import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import os

data = np.loadtxt('data.txt')

# X = All columns except the last one (the 1404 landmarks)
# y = The very last column (the emotion index)
X = data[:, :-1]
y = data[:, -1]

# 2. Split into Training (80%) and Testing (20%) sets
# stratify=y ensures we have an equal mix of all 7 emotions in both sets
X_train, X_test, y_train, y_test = train_test_split( X,
                                                     y,
                                                       test_size=0.2,
                                                         random_state=42,
                                                           stratify=y)

# 3. Initialize and Train the Random Forest Classifier
print(f"Training Random Forest on {len(X_train)} samples...")

# n_estimators=100: Use 100 decision trees
# max_depth: Limits how deep the trees go to prevent overfitting
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"📊 Accuracy: {accuracy * 100:.2f}%")

with open('emotion_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("💾 Model saved as 'emotion_model.pkl'")
print("📋 Classification Report:")
print(classification_report(y_test, y_pred))
print("🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))