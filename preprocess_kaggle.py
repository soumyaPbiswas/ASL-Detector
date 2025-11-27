import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "asl_alphabet_train"

os.makedirs("data", exist_ok=True)
mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

ignore = ["J", "Z"]
X = []
y = []

label_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d not in ignore])

for label in label_names:
    label_dir = os.path.join(DATA_DIR, label)
    files = [f for f in os.listdir(label_dir) if f.lower().endswith((".jpg",".png",".jpeg"))]
    for f in files:
        p = os.path.join(label_dir, f)
        img = cv2.imread(p)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r = mp_hands.process(img_rgb)
        if not r.multi_hand_landmarks:
            continue
        lm = []
        hand = r.multi_hand_landmarks[0]
        for l in hand.landmark:
            lm.append(l.x)
            lm.append(l.y)
        lm = np.array(lm)
        xs = lm[0::2]
        ys = lm[1::2]
        lm[0::2] = (xs - xs[0]) / (xs.max() - xs.min() + 1e-6)
        lm[1::2] = (ys - ys[0]) / (ys.max() - ys.min() + 1e-6)
        X.append(lm)
        y.append(label)

X = np.array(X, dtype=np.float32)
le = LabelEncoder()
Y = le.fit_transform(y)
labels = le.classes_

np.save("data/X.npy", X)
np.save("data/y.npy", Y)
np.save("data/labels.npy", labels)
