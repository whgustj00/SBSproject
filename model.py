import datetime
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n2.pt")

print(model.names)