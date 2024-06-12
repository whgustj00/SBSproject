import datetime
from ultralytics import YOLO
import cv2
import numpy as np
from shapely.geometry import Polygon

# 몇 가지 상수 정의
CONFIDENCE_THRESHOLD = 0.5  # 신뢰도 임계값
GREEN = (0, 255, 0)
RED = (0, 0, 255)  # 빨간색
YELLOW = (0, 255, 255)  # 노란색

# 모든 영역들의 점들을 저장하는 리스트
drawn_areas = []  # 모든 영역의 좌표를 저장함
color_switch = 0  # 영역색 결정을 위한 변수
new = True # 영역을 생성할 것인지 결정하는 변수

# 비디오 캡처 객체 초기화 (0: 기본 웹캠)
video_cap = cv2.VideoCapture(0)  # 0번 웹캠 / 1번 OBS 가상카메라

# 원본 프레임 크기 가져오기
ret, frame = video_cap.read()
if not ret:
    raise ValueError("웹캠을 열 수 없습니다.")
orig_height, orig_width = frame.shape[:2]

# 창 크기 설정
window_width, window_height = 1280, 720

def draw_area(event, x, y, flags, param):
    global drawn_areas, area_color, color_switch, new
    global orig_width, orig_height, window_width, window_height

    # 화면 좌표를 원본 프레임 좌표로 변환
    x_orig = int(x * orig_width / window_width)
    y_orig = int(y * orig_height / window_height)

    if event == cv2.EVENT_LBUTTONDOWN:
        # 현재 가상 영역의 색상 선택
        if color_switch == 0:
            area_color = YELLOW  # 노란색
        else:
            area_color = RED  # 빨간색

        if not drawn_areas or new == True:
            drawn_areas.append([area_color, (x_orig, y_orig)])  # 새로운 영역 시작
            new = False
        else:
            drawn_areas[-1].append((x_orig, y_orig))  # 현재 영역에 점 추가

    if event == cv2.EVENT_RBUTTONDOWN:
        new = True # 오른쪽 버튼 클릭으로 영역 완성

# 선과 바운딩 박스 10%가 교차하는지 확인하는 함수
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

# 사전 훈련된 YOLOv8n 모델 로드
model = YOLO("yolov8m.pt")

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Frame", draw_area)

while True:
    # FPS 계산을 위한 시작 시간
    start = datetime.datetime.now()

    ret, frame = video_cap.read()
    if not ret:
        break

    # 프레임에서 YOLO 모델 실행 (가상선을 그리기 전 프레임 사용)
    detections = model(frame)[0]

    # 감지된 객체에 대해 반복
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])

        # 사람이 아닌 경우 바운딩 박스를 항상 초록색으로 그리기
        if class_id != 0 and (class_id == 2 or class_id == 6):
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            continue

        # 사람인 경우 가상영역과 교차 여부 확인 (하단 25%)
        if class_id == 0:
            object_color = GREEN  # 기본 색상은 초록색
            for idx, area in enumerate(drawn_areas):
                if len(area) >= 4:  # 완료된 영역만 확인
                    if check_area_overlap(area, (xmin, ymin, xmax, ymax)):
                        object_color = area[0]
                        break
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), object_color, 2)

    # 영역 그리기 (감지 후 프레임에 그림)
    for idx, area in enumerate(drawn_areas):
        if len(area) >= 3:
            pts = np.array(area[1:], np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=area[0], thickness=2)

    # FPS 계산을 위한 끝 시간
    end = datetime.datetime.now()
    # 1프레임 처리에 걸린 시간 표시
    total = (end - start).total_seconds()
    print(f"1프레임 처리에 걸린 시간: {total * 1000:.0f} 밀리초")

    # FPS 계산 및 프레임에 표시
    fps = f"FPS: {1 / total:.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # 원본 프레임을 사용하여 창 크기와 프레임 크기 동시에 조정하여 표시
    frame_resized = cv2.resize(frame, (window_width, window_height))
    cv2.imshow("Frame", frame_resized)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("프로그램을 종료합니다.")
        break
    elif key == ord('r'):
        color_switch = 1
    elif key == ord('y'):
        color_switch = 0
    elif key == ord('t'):
        drawn_areas.clear()
    elif key == ord('u'):
        drawn_areas.pop(-1)

video_cap.release()
cv2.destroyAllWindows()
