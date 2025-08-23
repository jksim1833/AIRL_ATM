import cv2
import numpy as np
import math

def disck(img, rect_center_x, rect_center_y, circle_radius, scale):
    """
    첫 번째 함수: 큰 원 + 큰 원 위에 위치한 작은 원 + 선
    """
    # 큰 원 그리기
    cv2.circle(img, (rect_center_x, rect_center_y), circle_radius, (0, 0, 255), 2)
    
    # 원 안에 가로로 가로지르는 선 그리기 (원의 반지름과 동일한 길이)
    line_thickness = 3
    line_half_length = circle_radius
    line_start_x = rect_center_x - line_half_length
    line_end_x = rect_center_x + line_half_length
    cv2.line(img, (line_start_x, rect_center_y), (line_end_x, rect_center_y), (0, 255, 0), line_thickness)
    
    # 원의 제일 상단에 작은 원 그리기
    small_circle_radius = int(0.3 * scale / 2)
    top_circle_center_x = rect_center_x
    top_circle_center_y = rect_center_y - circle_radius  # 원의 제일 상단
    cv2.circle(img, (top_circle_center_x, top_circle_center_y), small_circle_radius, (255, 0, 255), -1)

def seat(img, rect_center_x, rect_center_y, scale):
    """
    두 번째 함수: 삼각형 + 삼각형 위에 위치한 작은 원
    """
    triangle_size = int(1.5 * scale)
    triangle_center_x = rect_center_x
    
    # 삼각형의 높이 계산
    triangle_height = int(triangle_size * math.sqrt(3) / 2)
    
    # 삼각형의 북쪽 꼭짓점에서 2/3 지점이 선(rect_center_y)에 오도록 위치 계산
    triangle_top_y = rect_center_y - (triangle_height * 2 // 3)
    triangle_bottom_y = triangle_top_y + triangle_height
    
    # 위쪽 꼭짓점 (북쪽 방향)
    top_point = (triangle_center_x, triangle_top_y)
    # 왼쪽 아래 꼭짓점
    left_point = (triangle_center_x - triangle_size // 2, triangle_bottom_y)
    # 오른쪽 아래 꼭짓점
    right_point = (triangle_center_x + triangle_size // 2, triangle_bottom_y)
    
    # 삼각형 그리기
    triangle_points = np.array([top_point, left_point, right_point], np.int32)
    triangle_points = triangle_points.reshape((-1, 1, 2))
    cv2.polylines(img, [triangle_points], True, (255, 0, 0), 2)
    
    # 북쪽 꼭짓점에 작은 원 그리기
    small_circle_radius = int(0.3 * scale / 2)
    cv2.circle(img, top_point, small_circle_radius, (255, 0, 255), -1)

def create_scene():
    """메인 장면 생성 함수"""
    # 스케일 팩터 (해상도 업스케일링)
    scale = 80
    
    # 캔버스 크기 설정 (정사각형보다 여유있게)
    canvas_width = int(12 * scale)
    canvas_height = int(12 * scale)
    
    # 흰색 배경 생성
    img = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # 1. 5*5 크기의 정사각형 그리기
    rect_width = int(5 * scale)
    rect_height = int(5 * scale)
    rect_x = (canvas_width - rect_width) // 2
    rect_y = (canvas_height - rect_height) // 2
    cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 0), 2)
    
    # 정사각형의 중심 좌표
    rect_center_x = rect_x + rect_width // 2
    rect_center_y = rect_y + rect_height // 2
    
    # 원의 반지름 계산
    circle_radius = int(4.5 * scale / 2)
    
    # 2. disck 함수 호출 (큰 원 + 큰 원 위 작은 원 + 선)
    disck(img, rect_center_x, rect_center_y, circle_radius, scale)
    
    # 3. seat 함수 호출 (삼각형 + 삼각형 위 작은 원)
    seat(img, rect_center_x, rect_center_y, scale)
    
    # 이미지 저장
    cv2.imwrite('opencv_structured.png', img)
    
    # 이미지 표시
    cv2.imshow('Structured OpenCV Drawing', img)
    
    print("구조화된 그림이 생성되었습니다!")
    print(f"캔버스 크기: {canvas_width} x {canvas_height}")
    print(f"정사각형 크기: {rect_width} x {rect_height} (5x5 단위)")
    print(f"원의 반지름: {circle_radius} (4.5/2 단위)")
    print(f"스케일: {scale}")
    print("\n구조:")
    print("- disck 함수: 큰 원 + 원 위 작은 원 + 가로선")
    print("- seat 함수: 삼각형 + 삼각형 위 작은 원")
    print("\n키보드의 아무 키나 누르면 창이 닫힙니다.")
    
    # 키 입력 대기 (창을 유지하기 위해)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return img

# 메인 실행 부분
if __name__ == "__main__":
    create_scene()