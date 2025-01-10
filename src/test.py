import cv2
import os
from tensorflow.keras.models import model_from_json
import numpy as np

json_file = open("trained model/handsign_model.json", 'r') 
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("trained model/handsign_model.h5")


cap = cv2.VideoCapture(0)

labels = ['G', 'H', 'K', 'O', 'Q', 'W', 'blank']

while True:
    ret, frame = cap.read()
 
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (0, 40), (300, 300) ,(255, 255, 255), 2)

    _frame = frame[40 : 300,0 : 300]
    _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)
    _frame = cv2.resize(_frame, (48, 48))
    _frame = np.array(_frame, dtype=np.float32)
    _frame = _frame.reshape(1, 48, 48, 1)
    _frame = _frame / 255.0
    predict = model.predict(_frame)
    predict_letter = labels[predict.argmax()]
   
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 200), -1)
    if predict_letter == 'blank':
        cv2.putText(frame, " ", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        accuracy = "{:.3f}".format(np.max(predict) * 100)
        cv2.putText(frame, f'{predict_letter}  {accuracy}%', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("output",frame)
    cv2.waitKey(25)
    
cap.release()
cv2.destroyAllWindows()