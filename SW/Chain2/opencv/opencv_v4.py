import cv2
import numpy as np
import math

def disck(img, rect_center_x, rect_center_y, circle_radius, scale, disk_type='W'):
    """
    첫 번째 함수: 큰 원 + 큰 원 위에 위치한 작은 원 + 선
    disk_type: 'W' -> 가로선 + 위쪽 작은 원, 'H' -> 세로선 + 우측 작은 원
    """
    if disk_type not in ('W', 'H'):
        raise ValueError("disk_type must be 'W' or 'H'")
    
    # 큰 원 그리기
    cv2.circle(img, (rect_center_x, rect_center_y), circle_radius, (0, 0, 255), 2)
    
    line_thickness = 3
    line_half_length = circle_radius
    small_circle_radius = int(0.3 * scale / 2)
    
    if disk_type == 'W':
        # W타입: 가로선 + 위쪽 작은 원 (기본 상태)
        line_start_x = rect_center_x - line_half_length
        line_end_x = rect_center_x + line_half_length
        cv2.line(img, (line_start_x, rect_center_y), (line_end_x, rect_center_y), (0, 255, 0), line_thickness)
        
        # 원의 제일 상단에 작은 원 그리기
        top_circle_center_x = rect_center_x
        top_circle_center_y = rect_center_y - circle_radius
        cv2.circle(img, (top_circle_center_x, top_circle_center_y), small_circle_radius, (255, 0, 255), -1)
        
    elif disk_type == 'H':
        # H타입: 세로선 + 우측 작은 원 (90도 회전된 상태)
        line_start_y = rect_center_y - line_half_length
        line_end_y = rect_center_y + line_half_length
        cv2.line(img, (rect_center_x, line_start_y), (rect_center_x, line_end_y), (0, 255, 0), line_thickness)
        
        # 원의 제일 우측에 작은 원 그리기
        right_circle_center_x = rect_center_x + circle_radius
        right_circle_center_y = rect_center_y
        cv2.circle(img, (right_circle_center_x, right_circle_center_y), small_circle_radius, (255, 0, 255), -1)

def seat(img, rect_center_x, rect_center_y, scale, disk_type='W', position_type='B'):
    """
    두 번째 함수: 삼각형 + 삼각형 위에 위치한 작은 원
    disk_type: 'W' 또는 'H' (첫 번째 함수의 선 방향 정보)
    position_type: 'A'(시작점), 'B'(중심), 'C'(종점)
    """
    if position_type not in ('A', 'B', 'C'):
        raise ValueError("position_type must be 'A', 'B', or 'C'")
    
    triangle_size = int(1.5 * scale)
    circle_radius = int(4.5 * scale / 2)  # 첫 번째 함수의 원 반지름과 동일
    
    # 위치에 따른 삼각형 중심 좌표 계산
    if disk_type == 'W':
        # W타입: 가로선 기준
        if position_type == 'A':
            # 시작점 (제일 좌측)
            triangle_center_x = rect_center_x - circle_radius
        elif position_type == 'B':
            # 중심점 (기본 위치)
            triangle_center_x = rect_center_x
        else:  # position_type == 'C'
            # 종점 (제일 우측)
            triangle_center_x = rect_center_x + circle_radius
        triangle_center_y = rect_center_y
        
    else:  # disk_type == 'H'
        # H타입: 세로선 기준
        triangle_center_x = rect_center_x
        if position_type == 'A':
            # 시작점 (제일 위쪽)
            triangle_center_y = rect_center_y - circle_radius
        elif position_type == 'B':
            # 중심점 (기본 위치)
            triangle_center_y = rect_center_y
        else:  # position_type == 'C'
            # 종점 (제일 아래쪽)
            triangle_center_y = rect_center_y + circle_radius
    
    # 삼각형의 높이 계산
    triangle_height = int(triangle_size * math.sqrt(3) / 2)
    
    # 삼각형의 북쪽 꼭짓점에서 2/3 지점이 선에 오도록 위치 계산
    triangle_top_y = triangle_center_y - (triangle_height * 2 // 3)
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

def create_scene(disk_type='W', position_type='B'):
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
    
    # 2. disck 함수 호출 (타입에 따른 큰 원 + 작은 원 + 선)
    disck(img, rect_center_x, rect_center_y, circle_radius, scale, disk_type)
    
    # 3. seat 함수 호출 (위치에 따른 삼각형 + 삼각형 위 작은 원)
    seat(img, rect_center_x, rect_center_y, scale, disk_type, position_type)
    
    # 이미지 저장
    filename = f'opencv_{disk_type}_{position_type}_type.png'
    cv2.imwrite(filename, img)
    
    # 이미지 표시
    window_name = f'{disk_type}{position_type} Type Drawing'
    cv2.imshow(window_name, img)
    
    # 위치 정보 텍스트
    position_names = {'A': '시작점', 'B': '중심', 'C': '종점'}
    disk_names = {'W': '가로선', 'H': '세로선'}
    
    print(f"{disk_type}{position_type}타입 구조화된 그림이 생성되었습니다!")
    print(f"파일명: {filename}")
    print(f"캔버스 크기: {canvas_width} x {canvas_height}")
    print(f"정사각형 크기: {rect_width} x {rect_height} (5x5 단위)")
    print(f"원의 반지름: {circle_radius} (4.5/2 단위)")
    print(f"스케일: {scale}")
    print(f"\n{disk_type}{position_type}타입 구조:")
    print(f"- disck 함수: 큰 원 + 작은 원 + {disk_names[disk_type]}")
    print(f"- seat 함수: 삼각형 + 작은 원 ({disk_names[disk_type]}의 {position_names[position_type]}에 위치)")
    print("\n키보드의 아무 키나 누르면 창이 닫힙니다.")
    
    # 키 입력 대기 (창을 유지하기 위해)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return img

# ===========================================
# 기본 구조화 코드 정보 (변화 적용시 참고용)
# ===========================================
# 스케일: 80 (1단위 = 80픽셀)
# 캔버스: 960x960 픽셀
# 정사각형: 400x400 픽셀 (5x5 단위)
# 큰 원 반지름: 180 픽셀 (4.5/2 단위) 
# 삼각형 크기: 120 픽셀 (1.5 단위)
# 작은 원 반지름: 12 픽셀 (0.3/2 단위)
# 선 두께: 3 픽셀
#
# 함수 구조:
# - disck(): 큰 원 + 원 위 작은 원 + 가로선
# - seat(): 삼각형 + 삼각형 위 작은 원  
# - create_scene(): 메인 장면 생성
#
# 색상:
# - 정사각형: (0, 0, 0) 검은색
# - 큰 원: (0, 0, 255) 빨간색
# - 가로선: (0, 255, 0) 녹색  
# - 삼각형: (255, 0, 0) 파란색
# - 작은 원들: (255, 0, 255) 자홍색
# ===========================================

# 메인 실행 부분
if __name__ == "__main__":
    # WA타입으로 실행 (가로선의 시작점에 삼각형 위치)
    create_scene('W', 'C')