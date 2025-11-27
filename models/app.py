import cv2
import mediapipe as mp
import numpy as np
import joblib
import streamlit as st

st.set_page_config(page_title="ASL Recognizer")

model = joblib.load("models/asl.pkl")
labels = np.load("models/labels.npy")

mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

st.title("ASL Recognizer")
run = st.checkbox("Start")
frame_window = st.image([])
cap = None

while run:
    if cap is None:
        cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r = mp_hands.process(rgb)
    pred = ""
    if r.multi_hand_landmarks:
        hand = r.multi_hand_landmarks[0]
        lm = []
        for l in hand.landmark:
            lm.append(l.x)
            lm.append(l.y)
        lm = np.array(lm, dtype=np.float32)
        xs = lm[0::2]
        ys = lm[1::2]
        lm[0::2] = (xs - xs[0]) / (xs.max() - xs.min() + 1e-6)
        lm[1::2] = (ys - ys[0]) / (ys.max() - ys.min() + 1e-6)
        lm = lm.reshape(1, -1)
        idx = model.predict(lm)[0]
        pred = labels[idx]
    frame = cv2.putText(frame, pred, (10,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    frame_window.image(frame, channels="BGR")
else:
    if cap:
        cap.release()
        cap = None
