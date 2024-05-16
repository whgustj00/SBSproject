import datetime
from ultralytics import YOLO
import cv2

# 몇 가지 상수 정의
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
RED = (0, 0, 255)  # 물체가 선에 닿았을 때 bounding box에 사용되는 빨간색
line_color = (0, 0, 255)  # 가상 선에 사용되는 빨간색
line_color2 = (0, 255, 255) # 노란색
# 모든 그린 선들의 점들을 저장하는 리스트
drawn_lines = []  # 모든 선 세그먼트(두 점의 튜플)를 저장함

# 마우스 이벤트를 처리하기 위한 콜백 함수
def draw_line(event, x, y, flags, param):
    global drawn_lines, line_color

    if event == cv2.EVENT_LBUTTONDOWN:
        # 현재 가상 선의 색상 선택
        if len(drawn_lines) % 2 == 0:
            line_color = (0, 255, 255)  # 노란색
        else:
            line_color = (0, 0, 255)  # 빨간색
        drawn_lines.append([(x, y)])  # 현재 점으로 새 선 시작

    elif event == cv2.EVENT_LBUTTONUP:
        if len(drawn_lines[-1]) == 1:  # 지금까지 점이 하나만 선택됐을 때
            drawn_lines[-1].append((x, y))  # 두 번째 점 추가
            cv2.line(frame, drawn_lines[-1][0], drawn_lines[-1][1], line_color, 2)


# 선과 점이 교차하는지 확인하는 함수
def check_line_intersection(point, line):
    """
    이 함수는 점이 선 세그먼트와 교차하는지 확인합니다.

    Args:
        point: (x, y)를 나타내는 튜플.
        line: 선 세그먼트를 두 점으로 나타내는 튜플 ((x1, y1), (x2, y2)).

    Returns:
        점이 선 세그먼트와 교차하면 True, 그렇지 않으면 False.
    """

    x1, y1 = line[0]
    x2, y2 = line[1]
    px, py = point

    # 수직선인지 확인
    if x1 == x2:
        return px == x1 and min(y1, y2) <= py <= max(y1, y2)

    # 수평선인지 확인
    elif y1 == y2:
        return py == y1 and min(x1, x2) <= px <= max(x1, x2)

    # 선의 기울기와 y절편 계산
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    # 점이 선 방정식 상에 있는지 확인
    if py == slope * px + intercept:
        # 점이 선 세그먼트 범위 내에 있는지 확인
        return min(x1, x2) <= px <= max(x1, x2)

    return False


# 비디오 캡처 객체 초기화 (0: 기본 웹캠)
video_cap = cv2.VideoCapture(1) # 0번 웹캠 / 1번 OBS 가상카메라

# 사전 훈련된 YOLOv8n 모델 로드
model = YOLO("yolov8n2.pt")

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", draw_line)

while True:
    # FPS 계산을 위한 시작 시간
    start = datetime.datetime.now()

    ret, frame = video_cap.read()

    # 이전에 그린 선들 그리기
    for idx, line in enumerate(drawn_lines):
        if len(line) == 2:  # 완료된 선만 확인
            if idx % 2 == 0:
                line_color = (0, 255, 255)  # 노란색
            else:
                line_color = (0, 0, 255)  # 빨간색
            cv2.line(frame, line[0], line[1], line_color, 2)

    # 프레임에서 YOLO 모델 실행
    detections = model(frame)[0]

    # 감지된 객체에 대해 반복
    for data in detections.boxes.data.tolist():
        # 감지와 연관된 신뢰도(확률) 추출
        confidence = data[4]

        # 최소 신뢰도보다 큰 신뢰도만 필터링
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # 감지된 객체가 사람인지 확인 (일반적으로 YOLO에서는 사람에 대해 클래스 인덱스 0을 사용함)
        if int(data[5]) != 0 and int(data[5]) != 2 and int(data[5]) != 6:  # 사람이 아님
            continue

        # 최소 신뢰도가 기준보다 큰 경우, bounding box 그리기
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        object_color = GREEN  # 객체 색상 초기화

        # 물체와 선 세그먼트 간의 충돌 확인
        for idx, line in enumerate(drawn_lines):
            if len(line) == 2:  # 완료된 선만 확인
                # bounding box 내의 임의의 점이 선 위에 있는지 확인
                for x in range(xmin, xmax + 1):
                    for y in range(ymin, ymax + 1):
                        if check_line_intersection((x, y), line):
                            if idx % 2 == 0:
                                object_color = (0, 255, 255)  # 노란색
                            else:
                                object_color = (0, 0, 255)  # 빨간색
                            break
                    if object_color != GREEN:
                        break
                if object_color != GREEN:
                    break

        # bounding box 그리기
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), object_color, 2)

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
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
cv2.destroyAllWindows()
