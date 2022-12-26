import cvlib as cv
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

#- load saved model
model = load_model('./mask_wearing_detection_model.h5')

#- activate camera
webcam = cv2.VideoCapture(0)


if not webcam.isOpened():
    print("Could not open webcam")
    exit()
    

while webcam.isOpened():
 
    
    status, frame = webcam.read()
    
    if not status:
        print("Could not read frame")
        exit()
 
    #- detecting face using cvlib 'detect_face' function
    face, confidence = cv.detect_face(frame)
 
    #- boxing face area
    for idx, f in enumerate(face):
        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        
        if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[0] and 0 <= endY <= frame.shape[0]:
            
            face_region = frame[startY:endY, startX:endX]
            
            face_region1 = cv2.resize(face_region, (224, 224), interpolation = cv2.INTER_AREA)
            
            x = img_to_array(face_region1)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            #- predict mask or no_mask
            prediction = model.predict(x)
 
            if prediction < 0.5:
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "No Mask ({:.2f}%)".format((1 - prediction[0][0])*100)
                cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
            else:
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "Mask ({:.2f}%)".format(prediction[0][0]*100)
                cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
    #- displaying output
    cv2.imshow("vedio mask detection", frame)
 
    #- If you want to stop, press "q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
webcam.release()
cv2.destroyAllWindows() 