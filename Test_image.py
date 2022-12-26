import cv2
import cvlib as cv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


#- load saved model
model = load_model('./mask_wearing_detection_model.h5')

img_path = './data/test_img/no_mask.jpg'

img = cv2.imread(img_path)

face, confidence = cv.detect_face(img)

for idx, f in enumerate(face):
    (startX, startY) = f[0], f[1]
    (endX, endY) = f[2], f[3]
    
    if 0 <= startX <= img.shape[1] and 0 <= endX <= img.shape[1] and 0 <= startY <= img.shape[0] and 0 <= endY <= img.shape[0]:
            
        face_region = img[startY:endY, startX:endX]    
        face_region1 = cv2.resize(face_region, (224, 224), interpolation = cv2.INTER_AREA)
            
        x = img_to_array(face_region1)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
            
        #- predict mask or no_mask
        prediction = model.predict(x)
 
        if prediction < 0.5:
            cv2.rectangle(img, (startX,startY), (endX,endY), (0,0,255), 2)
            Y = startY - 10 if startY - 10 > 10 else startY + 10
            text = "No Mask ({:.2f}%)".format((1 - prediction[0][0])*100)
            cv2.putText(img, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
        else:
            cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)
            Y = startY - 10 if startY - 10 > 10 else startY + 10
            text = "Mask ({:.2f}%)".format(prediction[0][0]*100)
            cv2.putText(img, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
    #- displaying output
cv2.imshow("img mask detection", img)

cv2.waitKey()
cv2.destroyAllWindows()