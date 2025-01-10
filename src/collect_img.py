import os

import cv2


DATA_DIR = './dataset'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 350

cap = cv2.VideoCapture(0)
for j in range(86, 89):
    if not os.path.exists(os.path.join(DATA_DIR, chr(j))):
        os.makedirs(os.path.join(DATA_DIR, chr(j)))

    print('Collecting data for class {}'.format(chr(j)))

    done = False
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame,(0,40),(300,300),(255,255,255),2)
        frame=frame[40:300,0:300]
        cv2.putText(frame, 'Press "Q" to cap screen', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 200
    while counter < dataset_size:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)
        frame = frame[40:300, 0:300]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize to 48x48
        resized_frame = cv2.resize(gray_frame, (48, 48))
        
        cv2.imshow('frame', resized_frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, chr(j), '{}.jpg'.format(counter)), resized_frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()

