import cv2
import cvlib as cv
 
#- activate camera
webcam = cv2.VideoCapture(0)
 
if not webcam.isOpened():
    print("Could not open webcam")
    exit()
    
 
sample_num = 0    
captured_num = 0
    
while webcam.isOpened():
    
    status, frame = webcam.read()
    sample_num = sample_num + 1
    
    if not status:
        break
 
    #- detecting face using cvlib 'detect_face' function
    face, confidence = cv.detect_face(frame)
    
    print(face)
    print(confidence)
 
    #- boxing face area
    for idx, f in enumerate(face):
        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
 
        #- setting mask or no_mask data
        if sample_num % 8  == 0:
            captured_num = captured_num + 1
            face_in_img = frame[startY:endY, startX:endX, :]
            #cv2.imwrite('./data/mask/mask'+str(captured_num)+'.png', face_in_img) #-- collect mask data
            cv2.imwrite('./data/no_mask/no_mask'+str(captured_num)+'.png', face_in_img) #-- collect no_mask data
 
 
    #- displaying output
    cv2.imshow("captured frames", frame)        
    
    #- If you want to stop, press "q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
webcam.release()
cv2.destroyAllWindows()   
