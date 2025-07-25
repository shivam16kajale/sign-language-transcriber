from sklearn.ensemble import VotingClassifier, RandomForestClassifier, StackingClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mediapipe as mp
import cv2
import time

import pickle
with open("model_03.pkl","rb") as f:
    model=pickle.load(f)

vid = cv2.VideoCapture(0)
hand = mp.solutions.hands
draw = mp.solutions.drawing_utils
hand_model = hand.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9
)

output_text = "" #save output of final prediction
last_prediction = "" # save the prediction 
prediction_start_time = 0 # start timer for new prediction
time_takes = 1.5  # wait for 1.5 seconds to confirm the prediction

while True:
    success, frame = vid.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_model.process(rgb)

    if result.multi_hand_landmarks:
        draw.draw_landmarks(frame, result.multi_hand_landmarks[0], hand.HAND_CONNECTIONS)
        hand_points = []
        for lm in result.multi_hand_landmarks[0].landmark:
            hand_points.extend([lm.x, lm.y, lm.z])

        if len(hand_points) == 63:
            hand_np = np.array(hand_points).reshape(1, 21, 3)
            wrist = hand_np[:, 0, :]
            normalized = hand_np - wrist[:, np.newaxis, :]
            flat = normalized.reshape(1, -1)
            prediction = model.predict(flat)[0]

          
            if prediction == last_prediction:
                timer = time.time() - prediction_start_time
                if timer > time_takes:

                    if prediction == 'BACKSPACE':
                        output_text = output_text[:-1]
                    elif prediction == 'SPACE':
                        output_text += ' '
                    else:
                        output_text += prediction

                    last_prediction = ""
                    prediction_start_time = 0
                    time.sleep(0.5)  
            else:
                last_prediction = prediction
                prediction_start_time = time.time()

            
            cv2.putText(frame, f"Sign: {prediction}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

  
    cv2.putText(frame, f"Text: {output_text}", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Hand Sign to Text", frame)
    if cv2.waitKey(1) & 255 == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()