import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

os.makedirs("models", exist_ok=True)

X = np.load("data/X.npy")
y = np.load("data/y.npy")
labels = np.load("data/labels.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = MLPClassifier(hidden_layer_sizes=(128,64), activation="relu", solver="adam", max_iter=1000)
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(acc)
print(confusion_matrix(y_test, pred))

joblib.dump(model, "models/asl.pkl")
np.save("models/labels.npy", labels)
