import cv2
import numpy as np
import winsound

# Comment out the line below to enable reading from a recorded video source
cap = cv2.VideoCapture('assets/video/flight_video_short.mp4')

# Comment out the line below to enable reading from a live video source
# Parameter 0 - from webcam, 1 - other attached camera source
#cap = cv2.VideoCapture(1)

# Parameters for varying the object detection
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
#check coco.names to find list of objects this system can I identify
searchObject = 'car'
enableSound = True
alertType = 'assets/sound/alert2'

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# faster detection lower accuracy
#tiny lib
#modelConfiguration = 'yolo3-tiny.cfg'
#modelWeights = 'yolov3-tiny.weights'

# more accurate but slower speed
modelConfiguration = 'yolov3-320.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2] * wT), int(det[3] * hT)
                x,y = int((det[0] * wT) - w/2), int((det[1] * hT) - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indeces = cv2.dnn.NMSBoxes(bbox,confs,confidence,nmsThreshold)

    for i in indeces:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        if(classNames[classIds[i]] == searchObject):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(searchObject + ' object has been detected')
            if(enableSound):
                winsound.PlaySound(alertType, winsound.SND_FILENAME)
        else:
            cv2.rectangle(img, (x,y), (x+w,y+h),(255,0,255),2)
            cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)




while(True):
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img,1/255,(whT, whT), [0,0,0], crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()

    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    #print(outputNames)

    outputs = net.forward(outputNames)

    findObjects(outputs,img)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
