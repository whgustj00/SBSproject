import datetime
from ultralytics import YOLO
import cv2

# 몇 가지 상수 정의
GREEN = (0, 255, 0)
RED = (0, 0, 255)  # 빨간색
YELLOW = (0, 255, 255)  # 노란색

# 모든 그린 선들의 점들을 저장하는 리스트
drawn_lines = []  # 모든 선 세그먼트(두 점의 튜플)를 저장함
color_switch = 0  # 선 색 결정을 위한 변수

# 마우스 이벤트를 처리하기 위한 콜백 함수
def draw_line(event, x, y, flags, param):
    global drawn_lines, line_color
    global color_switch

    if event == cv2.EVENT_LBUTTONDOWN:
        # 현재 가상 선의 색상 선택
        if color_switch == 0:
            line_color = YELLOW  # 노란색
        else:
            line_color = RED  # 빨간색
        drawn_lines.append([line_color, (x, y)])

    elif event == cv2.EVENT_LBUTTONUP:
        if len(drawn_lines[-1]) == 2:  # 지금까지 점이 하나만 선택됐을 때
            drawn_lines[-1].append((x, y))  # 두 번째 점 추가

# 선과 바운딩 박스 하단 25%가 교차하는지 확인하는 함수
def check_line_box_bottom_intersection(line, box):
    """
    이 함수는 선과 바운딩 박스 하단 25%가 교차하는지 확인합니다.

    Args:
        line: 선 세그먼트를 두 점으로 나타내는 튜플 ((x1, y1), (x2, y2)).
        box: 바운딩 박스를 네 점으로 나타내는 튜플 (xmin, ymin, xmax, ymax).

    Returns:
        선과 바운딩 박스 하단 25%가 교차하면 True, 그렇지 않으면 False.
    """
    def line_intersects_line(p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    x1, y1 = line[1]
    x2, y2 = line[2]
    xmin, ymin, xmax, ymax = box

    # 바운딩 박스의 하단 25% 영역 계산
    y_bottom_start = ymin + int(0.75 * (ymax - ymin))
    y_bottom_end = ymax

    # 하단 25% 영역의 네 개 변
    bottom_box_lines = [
        ((xmin, y_bottom_start), (xmax, y_bottom_start)),
        ((xmax, y_bottom_start), (xmax, y_bottom_end)),
        ((xmax, y_bottom_end), (xmin, y_bottom_end)),
        ((xmin, y_bottom_end), (xmin, y_bottom_start))
    ]

    for box_line in bottom_box_lines:
        if line_intersects_line((x1, y1), (x2, y2), box_line[0], box_line[1]):
            return True
    return False

# 비디오 캡처 객체 초기화 (0: 기본 웹캠)
video_cap = cv2.VideoCapture(1)  # 0번 웹캠 / 1번 OBS 가상카메라

# 사전 훈련된 YOLOv8n 모델 로드
model = YOLO("yolov8m.pt")

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", draw_line)

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
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])

        # 사람이 아닌 경우 바운딩 박스를 항상 초록색으로 그리기
        if class_id != 0:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            continue

        # 사람인 경우 가상선과 교차 여부 확인 (하단 25%)
        object_color = GREEN  # 기본 색상은 초록색
        for idx, line in enumerate(drawn_lines):
            if len(line) == 3:  # 완료된 선만 확인
                if check_line_box_bottom_intersection(line, (xmin, ymin, xmax, ymax)):
                    object_color = line[0]
                    break

        # bounding box 그리기
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), object_color, 2)

    # 선 그리기 (감지 후 프레임에 그림)
    for idx, line in enumerate(drawn_lines):
        if len(line) == 3:
            cv2.line(frame, line[1], line[2], line[0], 2)
            print(line)
        else:
            pass

    # FPS 계산을 위한 끝 시간
    end = datetime.datetime.now()
    # 1프레임 처리에 걸린 시간 표시
    total = (end - start).total_seconds()
    print(f"1프레임 처리에 걸린 시간: {total * 1000:.0f} 밀리초")

    # FPS 계산 및 프레임에 표시
    fps = f"FPS: {1 / total:.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # 화면에 프레임 표시
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("프로그램을 종료합니다.")
        break
    elif key == ord('r'):
        color_switch = 1
    elif key == ord('y'):
        color_switch = 0
    elif key == ord('t'):
        drawn_lines.clear()

video_cap.release()
cv2.destroyAllWindows()
