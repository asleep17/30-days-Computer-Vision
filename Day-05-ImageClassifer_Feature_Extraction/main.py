import numpy as np
import pickle
import os
from PIL import Image
from img2vec_pytorch import Img2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Loading Functions
def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

def load_cifar10_train(path):
    images, labels = [], []
    for i in range(1, 6):
        batch = unpickle(os.path.join(path, f"data_batch_{i}"))
        X = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        images.append(X)
        labels.extend(batch[b'labels'])
    return np.vstack(images), np.array(labels)

# 2. Setup
data_path = "./cifar-10 data"
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
img2vec = Img2Vec(model='resnet18', cuda=False)

# 3. Batch Extraction Function (The Memory Saver)
def extract_safely(img_list, batch_size=32):
    features_list = []
    for i in range(0, len(img_list), batch_size):
        batch = img_list[i : i + batch_size]
        feats = img2vec.get_vec(batch)
        features_list.append(feats)
        if (i + batch_size) % 128 == 0 or (i + batch_size) >= len(img_list):
            print(f"Extracted: {min(i + batch_size, len(img_list))}/{len(img_list)}")
    return np.vstack(features_list)

# 4. Training Phase
print("Loading 2000 training images...")
X_train_raw, y_train_full = load_cifar10_train(data_path)
subset_size = 2000
train_images_pil = [Image.fromarray(img.astype('uint8')) for img in X_train_raw[:subset_size]]

print("Starting Training Feature Extraction (Batch Mode)...")
train_features = extract_safely(train_images_pil)

clf = LogisticRegression(max_iter=1000, n_jobs=-1)
clf.fit(train_features, y_train_full[:subset_size])

# SAVE THE MODEL IMMEDIATELY
with open('cifar10_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)
print("\nModel saved successfully as 'cifar10_classifier.pkl'")

# 5. Testing Phase
print("\nLoading test data...")
test_batch = unpickle(os.path.join(data_path, "test_batch"))
X_test_raw = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
y_test_full = np.array(test_batch[b'labels'])

test_subset = 500
test_images_pil = [Image.fromarray(img.astype('uint8')) for img in X_test_raw[:test_subset]]

print("Starting Test Feature Extraction (Batch Mode)...")
test_features = extract_safely(test_images_pil)

# 6. Report
y_pred = clf.predict(test_features)
print("\n--- CLASSIFICATION REPORT (2000 Training Samples) ---")
print(classification_report(y_test_full[:test_subset], y_pred, target_names=classes))