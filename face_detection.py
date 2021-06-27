import cv2

cap = cv2.VideoCapture(0)
facecascade = cv2.CascadeClassifier("/home/aman/.local/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml")

while True:
  isTrue,img = cap.read()
  img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  faces = facecascade.detectMultiScale(img_grey,1.1,4)
  for x,y,w,h in faces:
    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.circle(img,(x+(w//2),y+(h//2)),h//2,(0,255,0),2)
    
  cv2.imshow("video",img)
  if cv2.waitKey(20) == ord('q'):
    cap.release()
    cv2.destroyAllWindows()
    break

    

