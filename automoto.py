import cv2
import math
import random
import numpy as np
from ultralytics import YOLO


model = YOLO("yolo11m.pt")
cap = cv2.VideoCapture('cvtest_cut.avi')

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter('output.avi', fourcc, 25, (frame_width, frame_height))

#Зона обнаружения мащин (car) и мотоциклов (motorcycle) 
start_point = (607, 384)
end_point = (1627, 1384)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error occurred.")
        break

    crop_frame = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]
    results = model(source=crop_frame, conf=0.45, iou=0.5, show=False, save=False, verbose=False, classes=[2,3]) # car, motocycle
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    confidenes = np.around(results[0].boxes.conf.cpu().numpy().astype(float), decimals=2)

    for box, clss, confd in zip(boxes, classes, confidenes):
        if clss != 0:
            random.seed(int(clss))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) #Каждый класс имеет свой цвет
            for box in boxes:
                if (math.hypot(end_point[0] - start_point[0], end_point[1] - start_point[1])/3 < math.hypot(box[2] - box[0], box[3] - box[1])): #Если объект больше 1/3 зоны обнаружения
                    cv2.rectangle(frame, (start_point[0] + box[0], start_point[1] + box[1]), (start_point[0] + box[2], start_point[1] + box[3],), color, 2)
                    cv2.putText(frame,f"Class {model.names[clss]} , confidence {confd}",(start_point[0] + box[0], start_point[1] + box[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, )
    cv2.rectangle(frame, (start_point), (end_point), (0,255,0), 2)

    resize_frame = cv2.resize(frame, (int(frame_width/2), int(frame_height/2))) 
    cv2.imshow('video', resize_frame)
    
    out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()