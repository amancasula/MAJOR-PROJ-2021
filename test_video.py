import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
classes = []

with open("coco.names","r") as f:
  classes = [line.strip() for line in f.readlines()]

#print(classes)

layerNames = net.getLayerNames()

#print(layerNames)
#print(len(layerNames))

unconnected = net.getUnconnectedOutLayers()

#print(unconnected)
#print(type(unconnected))

output_layers = [layerNames[i[0]-1] for i in unconnected]

#print(output_layers)

cap = cv2.VideoCapture(0)

while True:
  istrue , img = cap.read()
  height, width, channels = img.shape

  #print(img.shape)

  blob = cv2.dnn.blobFromImage(img,0.00392,(416, 416), (0, 0, 0), True, crop=False)
  #print(blob)
  #print(len(blob),len(blob[0]),len(blob[0][0]),len(blob[0][0][0]))


  #cv2.imshow("1",blob[0][0])
  #cv2.imshow("2",blob[0][1])
  #cv2.imshow("3",blob[0][2])
  #cv2.waitKey(0) 
    
    
  net.setInput(blob)
  outs = net.forward(output_layers)

  #print(len(outs))
  #print(len(outs[0]))
  #print(len(outs[0][0]))
  #print(outs)

  boxes =[]
  confidances =[]
  labels = []

  for out in outs:
    for detection in out:
      scores = detection[5:]
      class_id = np.argmax(scores)
      confidance = scores[class_id]
      if confidance > 0.5:
        center_x = int(detection[0]*width)
        center_y = int(detection[1]*height)
        w = int(detection[2]*width)
        h = int(detection[3]*height)
        #print(classes[class_id])
      
        #cv2.circle(img,(center_x,center_y),10,(0,255,0),-1)
      
        x = int(center_x - w/2)
        y = int(center_y - h/2)
      
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
      
        confidances.append(float(confidance))
        boxes.append([x,y,w,h])
        labels.append(classes[class_id])
      
  indexes = cv2.dnn.NMSBoxes(boxes,confidances,0.5,0.4)

  for i in range(len(boxes)):
    if i in indexes:
      x,y,w,h = boxes[i]
      cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
      print(labels[i])
    
    

  cv2.imshow("img",img)
  if cv2.waitKey(20)  == ord('q'):
    cap.release()
    cv2.destroyAllWindows()
    break


