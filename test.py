import datetime
from ultralytics import YOLO
import cv2

# define some constants
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
line_color = (255, 0, 0)  # Red color for the virtual line

# List to store the points of all drawn lines
drawn_lines = []  # This will store all the line segments (tuples of two points)

# Mouse callback function to handle mouse events
def draw_line(event, x, y, flags, param):
    global drawn_lines

    if event == cv2.EVENT_LBUTTONDOWN:
        drawn_lines.append([(x, y)])  # Start a new line with the current point

    elif event == cv2.EVENT_LBUTTONUP:
        if len(drawn_lines[-1]) == 1:  # If only one point has been clicked so far
            drawn_lines[-1].append((x, y))  # Add the second point
            cv2.line(frame, drawn_lines[-1][0], drawn_lines[-1][1], line_color, 2)

# initialize the video capture object (0: default webcam)
video_cap = cv2.VideoCapture(0)

# load the pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", draw_line)

while True:
    # start time to compute the fps
    start = datetime.datetime.now()

    ret, frame = video_cap.read()

    # run the YOLO model on the frame
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

    # Draw all the drawn lines
    for line in drawn_lines:
        if len(line) == 2:
            cv2.line(frame, line[0], line[1], line_color, 2)

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

video_cap.release()
cv2.destroyAllWindows()
