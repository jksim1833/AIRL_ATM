import cv2
import numpy as np
import math

# 스케일 팩터 (적절한 확대를 위해)
scale = 40

# 캔버스 크기 설정 (직사각형보다 여유있게)
canvas_width = int(12 * scale)
canvas_height = int(12 * scale)

# 흰색 배경 생성
img = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

# 1. 5*10 크기의 직사각형 그리기
rect_width = int(5 * scale)
rect_height = int(10 * scale)
rect_x = (canvas_width - rect_width) // 2
rect_y = (canvas_height - rect_height) // 2
cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 0), 2)

# 직사각형의 중심 좌표
rect_center_x = rect_x + rect_width // 2
rect_center_y = rect_y + rect_height // 2

# 2. 4.5*4.5 크기의 원을 직사각형 정 가운데에 그리기
circle_radius = int(4.5 * scale / 2)
cv2.circle(img, (rect_center_x, rect_center_y), circle_radius, (0, 0, 255), 2)

# 3. 원 안에 가로로 가로지르는 선 그리기
line_thickness = 3
line_start_x = rect_center_x - circle_radius + 10
line_end_x = rect_center_x + circle_radius - 10
cv2.line(img, (line_start_x, rect_center_y), (line_end_x, rect_center_y), (0, 255, 0), line_thickness)

# 4. 선 위에 삼각형 그리기
triangle_size = int(1.5 * scale)
triangle_center_x = rect_center_x
triangle_center_y = rect_center_y

# 삼각형의 꼭짓점 계산 (북쪽을 가리키는 정삼각형)
# 높이 = (√3/2) * 한 변의 길이
triangle_height = int(triangle_size * math.sqrt(3) / 2)

# 위쪽 꼭짓점 (북쪽 방향)
top_point = (triangle_center_x, triangle_center_y - triangle_height // 2)
# 왼쪽 아래 꼭짓점
left_point = (triangle_center_x - triangle_size // 2, triangle_center_y + triangle_height // 2)
# 오른쪽 아래 꼭짓점
right_point = (triangle_center_x + triangle_size // 2, triangle_center_y + triangle_height // 2)

# 삼각형 그리기
triangle_points = np.array([top_point, left_point, right_point], np.int32)
triangle_points = triangle_points.reshape((-1, 1, 2))
cv2.polylines(img, [triangle_points], True, (255, 0, 0), 2)

# 4-4. 북쪽 꼭짓점에 0.3*0.3 크기의 원 그리기
small_circle_radius = int(0.3 * scale / 2)
cv2.circle(img, top_point, small_circle_radius, (255, 0, 255), -1)  # -1은 채우기

# 이미지 저장
cv2.imwrite('opencv_drawing_result.png', img)

# 이미지 표시 (선택사항 - 주석 처리됨)
cv2.imshow('OpenCV Drawing', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("그림이 성공적으로 생성되었습니다!")
print(f"캔버스 크기: {canvas_width} x {canvas_height}")
print(f"직사각형 크기: {rect_width} x {rect_height}")
print(f"원의 반지름: {circle_radius}")
print(f"삼각형 크기: {triangle_size}")
print(f"작은 원의 반지름: {small_circle_radius}")