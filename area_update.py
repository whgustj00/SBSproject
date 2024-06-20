import sys
import cv2
import numpy as np
import datetime
from shapely.geometry import Polygon
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QLabel, QLineEdit, QScrollArea, QDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO

# Constants
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold
GREEN = (0, 255, 0)
RED = (0, 0, 255)  # Red color
YELLOW = (0, 255, 255)  # Yellow color

# List to store the points of all areas
drawn_areas = []  # Coordinates of all areas
color_switch = 0  # Variable to determine area color
new = True  # Variable to determine whether to create a new area

# Video capture object initialization (0: default webcam)
video_cap = cv2.VideoCapture(0)  # 0 for default webcam

# Original frame size
ret, frame = video_cap.read()
if not ret:
    raise ValueError("Cannot open webcam.")
orig_height, orig_width = frame.shape[:2]

# Window size
window_width, window_height = 1280, 720

def draw_area(event, x, y, flags, param):
    global drawn_areas, area_color, color_switch, new
    global orig_width, orig_height, window_width, window_height

    # Convert screen coordinates to original frame coordinates
    x_orig = int(x * orig_width / window_width)
    y_orig = int(y * orig_height / window_height)

    if event == cv2.EVENT_LBUTTONDOWN:
        # Select the color of the current virtual area
        if color_switch == 0:
            area_color = YELLOW  # Yellow
        else:
            area_color = RED  # Red

        if not drawn_areas or new:
            drawn_areas.append([area_color, (x_orig, y_orig)])  # Start a new area
            new = False
        else:
            drawn_areas[-1].append((x_orig, y_orig))  # Add point to current area

    if event == cv2.EVENT_RBUTTONDOWN:
        new = True  # Complete area on right click

# Function to check if line and bounding box overlap by 10%
def check_area_overlap(area, box):  
    xmin, ymin, xmax, ymax = box

    poly1 = Polygon([i for i in area[1:]])

    height = ymax - ymin
    bottom_y = ymin + 0.75 * height

    rect2 = Polygon([(xmin, bottom_y), (xmax, bottom_y), 
                     (xmax, ymax), (xmin, ymax)])

    intersection = poly1.intersection(rect2)
    overlap_area = intersection.area
    rect2_area = rect2.area

    return overlap_area >= 0.50 * rect2_area

# Pre-trained YOLOv8 model
model = YOLO("yolov8x.pt")

class FrameProcessor(QThread):

    def __init__(self):
        super().__init__()

    new_frame = pyqtSignal(np.ndarray)

    def run(self):
        while True:
            ret, frame = video_cap.read()
            if not ret:
                break

            # Run YOLO model on the frame
            detections = model(frame)[0]

            # Iterate over detected objects
            for data in detections.boxes.data.tolist():
                confidence = data[4]
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                class_id = int(data[5])

                # Draw bounding box for non-human objects in green
                if class_id != 0 and (class_id == 2 or class_id == 6):
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 1)
                    continue

                # Check overlap with virtual area for human objects
                if class_id == 0:
                    object_color = GREEN  # Default color is green
                    for idx, area in enumerate(drawn_areas):
                        if len(area) >= 4:  # Check only completed areas
                            if check_area_overlap(area, (xmin, ymin, xmax, ymax)):
                                object_color = area[0]
                                if area[0] == RED:
                                    ex.add_log("DANGER")
                                elif area[0] == YELLOW:
                                    ex      .add_log("WARNING")
                            
                                break
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), object_color, 1)

            # Draw areas
            for idx, area in enumerate(drawn_areas):
                if len(area) >= 3:
                    pts = np.array(area[1:], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], isClosed=True, color=area[0], thickness=1)

            # Resize frame
            frame_resized = cv2.resize(frame, (window_width, window_height))

            # Emit new frame signal
            self.new_frame.emit(frame_resized)
            self.msleep(30)

class AreaSettingsDialog(QDialog):
    def __init__(self):
        super().__init__()

    def exec_(self):
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Frame", draw_area)
        
        while True:
            ret, frame = video_cap.read()
            if not ret:
                break

            # Draw areas
            for idx, area in enumerate(drawn_areas):
                if len(area) >= 3:
                    pts = np.array(area[1:], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], isClosed=True, color=area[0], thickness=1)

            # Resize frame
            frame_resized = cv2.resize(frame, (window_width, window_height))
            cv2.imshow("Frame", frame_resized)

            key = cv2.waitKey(1) & 0xFF
            global color_switch
            if key == ord('q'):
                break
            elif key == ord('r'):
                color_switch = 1
            elif key == ord('y'):
                color_switch = 0
            elif key == ord('t'):
                drawn_areas.clear()
            elif key == ord('u'):
                drawn_areas.pop(-1)

        cv2.destroyAllWindows()
        self.close()

class MyApp(QWidget):

    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.logs = []
        self.initUI()
        self.log_signal.connect(self.update_log)

        self.frame_processor = FrameProcessor()
        self.frame_processor.new_frame.connect(self.update_frame)
        self.frame_processor.start()

    def initUI(self):
        # Main layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Sidebar layout
        sidebar = QVBoxLayout()
        cam_list = QListWidget()
        cams = ['cam1', 'cam2', 'cam3', 'cam4']
        for cam in cams:
            cam_list.addItem(cam)
        sidebar.addWidget(cam_list)

        buttons = QVBoxLayout()
        settings_button = QPushButton('Settings')
        settings_button.clicked.connect(self.open_settings_dialog)
        buttons.addWidget(settings_button)
        for _ in range(2):
            button = QPushButton('Button')
            buttons.addWidget(button)
        sidebar.addLayout(buttons)

        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar)
        sidebar_widget.setStyleSheet("background-color: #333; color: #fff;")

        main_layout.addWidget(sidebar_widget, 1)

        # Main area layout
        main_area = QVBoxLayout()

        self.camera_view = QLabel()
        self.camera_view.setFixedSize(1280, 720)
        self.camera_view.setStyleSheet("border: 1px solid #ccc; background-color: #fff;")
        self.camera_view.setAlignment(Qt.AlignCenter)
        main_area.addWidget(self.camera_view)

        # 이벤트 로그
        event_logs = QVBoxLayout()
        self.log_content = QVBoxLayout()
        self.log_widget = QWidget()
        self.log_widget.setLayout(self.log_content)
        self.log_window = QScrollArea()
        self.log_window.setWidget(self.log_widget)
        self.log_window.setWidgetResizable(True)
        event_logs.addWidget(self.log_window)

        event_logs_widget = QWidget()
        event_logs_widget.setLayout(event_logs)
        event_logs_widget.setStyleSheet("background-color: #333; color: #fff;")

        main_area.addWidget(event_logs_widget, 1)

        main_widget = QWidget()
        main_widget.setLayout(main_area)

        main_layout.addWidget(main_widget, 2)

        # Main window settings
        self.setWindowTitle('UI Layout')
        self.setGeometry(100, 100, 1600, 900)
        self.setLayout(main_layout)
        self.show()

    def update_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_view.setPixmap(QPixmap.fromImage(qt_image))

    def open_settings_dialog(self):
        dialog = AreaSettingsDialog()
        dialog.exec_()

    def add_log(self, event):
        self.log_signal.emit(event)

    def update_log(self, event):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"log {len(self.logs)+1}: {current_time} {event}"
        self.logs.append(log_entry)
        # UI 업데이트
        log_label = QLabel(log_entry)
        self.log_content.addWidget(log_label)
        self.log_widget.setLayout(self.log_content)
        self.log_window.setWidget(self.log_widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()    
    sys.exit(app.exec_())
