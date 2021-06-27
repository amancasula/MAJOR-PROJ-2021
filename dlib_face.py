import cv2
import dlib

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
  _bool,img = cap.read()
  img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

   
  
  faces = detector(img_gray)
  
  
  for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    
    #FACE DETECTION
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
    
    landmarks = predictor(img_gray,face)
    
    ##LANDMARKS
    for i in range(68):
      x = landmarks.part(i).x
      y = landmarks.part(i).y
      cv2.circle(img,(x,y),3,(0,0,255),-1)
    
    ##MOUTH OPEN  
    y_mouth_a = landmarks.part(51).y
    y_mouth_b = landmarks.part(57).y
    if y_mouth_b - y_mouth_a > 25:
      cv2.putText(img,"MOUTH OPEN",(300,100),cv2.FONT_ITALIC,1,(0,150,0),3)
    
    ##BLINKING
    y_eyes_a_l = landmarks.part(38).y
    y_eyes_b_l = landmarks.part(40).y
    y_eyes_a_r = landmarks.part(44).y
    y_eyes_b_r = landmarks.part(46).y
    if (y_eyes_b_l - y_eyes_a_l) <= 7 or (y_eyes_b_r - y_eyes_a_r) <= 7:
      cv2.putText(img,"BLINKED",(300,300),cv2.FONT_ITALIC,1,(0,150,0),3)
      
    
          
    
      
   
    cv2.imshow("video",img)
  if cv2.waitKey(20) == ord('q'):
    cap.release()
    cv2.destroyAllWindows()
    break
    
  
