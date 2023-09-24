import cv2
frame = cv2.imread("C:\\Users\\yeshwanthr\\Documents\\Jupiter nb python files\\Object_Detector\\Images\\Bottle\\test7.jpg")
frame = cv2.resize(frame,(640,640))
yolo = cv2.dnn.readNetFromONNX("C:\\Users\\yeshwanthr\\Documents\\Jupiter nb python files\\Object_Detector\\best.onnx")
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
blob = cv2.dnn.blobFromImage(frame,1/255,(640,640),swapRB=True,crop=False)
yolo.setInput(blob)
preds = yolo.forward()
print(preds.shape)
#centerX CenterY width Height confidence Prob-score-class1 prob-score-class2 ...
#confidence should be more than 0.4 and probability more than 0.25
boxes=[]
scores = []
conf = []
detections = preds[0]
for detection in detections:
    if(detection[4]>0.4 and detection[5]>0.5):
        x = int(detection[0]-(0.5*detection[2]))
        y = int(detection[1]-(0.5*detection[3]))
        w = int(detection[2])
        h = int(detection[3])
        boxes.append((x,y,w,h))
        scores.append(detection[5])
        conf.append(detection[4])
print(boxes)
print(conf)
print(scores)
indices = cv2.dnn.NMSBoxes(boxes, conf, 0.25, 0.5)
for index in indices:
    x,y,w,h = boxes[index]
    cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h),color=(0,0,255),thickness=1)
cv2.imshow("frame",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
