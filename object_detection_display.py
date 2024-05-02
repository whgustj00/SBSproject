import datetime
import cv2
import mss
import numpy as np
from ultralytics import YOLO

# define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)

# Initialize the mss instance
sct = mss.mss()

# Get monitor information
monitor = sct.monitors[1]  # Change the index to capture a different monitor if needed

# Define monitor capture area
monitor_capture_area = {
    "left": monitor["left"],
    "top": monitor["top"],
    "width": monitor["width"],
    "height": monitor["height"]
}

# Load the pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")

while True:
    # start time to compute the fps
    start = datetime.datetime.now()

    # Capture screen
    sct_img = sct.grab(monitor_capture_area)
    frame = np.array(sct_img)

    # Run the YOLO model on the frame
    detections = model(frame)[0]

    # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the detection
        confidence = data[4]

        # filter out weak detections by ensuring the 
        # confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # Check if the detection is a person (class index 0 is usually used for person in YOLO)
        if int(data[5]) != 0:  # Not a person
            continue

        # if the confidence is greater than the minimum confidence,
        # draw the bounding box on the frame
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)

    # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    total = (end - start).total_seconds()
    print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")

    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / total:.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
