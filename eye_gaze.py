import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap.set(3,640)
cap.set(4,480)


while True:
  mask = np.zeros((480,640),np.uint8)
  _,img = cap.read()
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  blur_gray = cv2.GaussianBlur(gray,(7,7),0)
  _,img_bw = cv2.threshold(gray,60,255,cv2.THRESH_BINARY_INV)
  faces = detector(gray)
  for face in faces:
    landmarks = predictor(gray,face)
    
    left_eye = np.array([(landmarks.part(36).x,landmarks.part(36).y),(landmarks.part(37).x,landmarks.part(37).y),(landmarks.part(38).x,landmarks.part(38).y),(landmarks.part(39).x,landmarks.part(39).y),(landmarks.part(40).x,landmarks.part(40).y),(landmarks.part(41).x,landmarks.part(41).y)],np.int32)
    
    cv2.polylines(mask,[left_eye],True,255,2)
    cv2.fillPoly(mask,[left_eye],255)
  
  
  and_img = cv2.bitwise_and(img_bw,img_bw,mask=mask)
  contours,_ = cv2.findContours(and_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  #print(len(contours))
  contours = sorted(contours,key=lambda x:cv2.contourArea(x),reverse = True) 
  #cv2.drawContours(img,[contours[0]],-1,(0,0,255),3)
  x,y,w,h = cv2.boundingRect(contours[0])
  xc,yc = x+(w//2) , y+(h//2)
  cv2.circle(img,(xc,yc),2,(0,0,255),-1) 
  p36_x = landmarks.part(36).x
  p39_x = landmarks.part(39).x
  #print("left:"+str(xc - p36_x))
  #print("right:"+str(xc - p39_x))
  left = xc - p36_x
  if left >= 26:
    cv2.putText(img,"LOOKING LEFT",(300,100),cv2.FONT_ITALIC,1,(255,0,0),1)
    print("LOOKING LEFT")
  elif left <= 10:
    cv2.putText(img,"LOOKING RIGHT",(300,100),cv2.FONT_ITALIC,2,(255,0,0),3)
    print("LOOKING RIGHT")
  
  cv2.imshow("video1",img)
  #cv2.imshow("mask",mask)
  #cv2.imshow("bw",img_bw)
  #cv2.imshow("and_img",and_img)
  
  if cv2.waitKey(20) == ord('q'):
    
    cap.release()
    cv2.destroyAllWindows()
    break


  
