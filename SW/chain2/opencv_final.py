import cv2
import numpy as np
import math

# ===========================================
# 설정 상수
# ===========================================
SCALE = 80                    # 해상도 스케일 (1단위 = 80픽셀)
CANVAS_SIZE = 12             # 캔버스 크기 (12 * SCALE)
RECT_SIZE = 5                # 정사각형 크기 (5 * SCALE)
CIRCLE_SIZE = 4.5            # 큰 원 크기 (4.5 * SCALE)
TRIANGLE_SIZE = 1.5          # 삼각형 크기 (1.5 * SCALE)
SMALL_CIRCLE_SIZE = 0.3      # 작은 원 크기 (0.3 * SCALE)

# 색상 정의 (BGR 형식)
COLORS = {
    'RECT': (0, 0, 0),           # 검은색 - 정사각형
    'CIRCLE': (0, 0, 255),       # 빨간색 - 큰 원
    'LINE': (0, 255, 0),         # 녹색 - 선
    'TRIANGLE': (255, 0, 0),     # 파란색 - 삼각형
    'SMALL_CIRCLE': (255, 0, 255) # 자홍색 - 작은 원
}

# 선 두께
LINE_THICKNESS = 3

# ===========================================
# 유틸리티 함수
# ===========================================
def validate_disk_type(disk_type):
    """disk_type 유효성 검사"""
    if disk_type not in ('W', 'H'):
        raise ValueError("disk_type must be 'W' or 'H'")

def validate_position_type(position_type):
    """position_type 유효성 검사"""
    if position_type not in ('A', 'B', 'C'):
        raise ValueError("position_type must be 'A', 'B', or 'C'")

def validate_direction_type(direction_type):
    """direction_type 유효성 검사"""
    if direction_type not in ('F', 'R', 'B', 'L'):
        raise ValueError("direction_type must be 'F', 'R', 'B', or 'L'")

def get_canvas_setup():
    """캔버스 기본 설정 반환"""
    canvas_width = int(CANVAS_SIZE * SCALE)
    canvas_height = int(CANVAS_SIZE * SCALE)
    return canvas_width, canvas_height

def get_rect_setup(canvas_width, canvas_height):
    """정사각형 설정 반환"""
    rect_width = int(RECT_SIZE * SCALE)
    rect_height = int(RECT_SIZE * SCALE)
    rect_x = (canvas_width - rect_width) // 2
    rect_y = (canvas_height - rect_height) // 2
    rect_center_x = rect_x + rect_width // 2
    rect_center_y = rect_y + rect_height // 2
    
    return rect_width, rect_height, rect_x, rect_y, rect_center_x, rect_center_y

def get_circle_radius():
    """원의 반지름 반환"""
    return int(CIRCLE_SIZE * SCALE / 2)

# ===========================================
# 그리기 함수
# ===========================================
def draw_rectangle(img, rect_x, rect_y, rect_width, rect_height):
    """정사각형 그리기"""
    cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), 
                  COLORS['RECT'], 2)

def disck(img, rect_center_x, rect_center_y, circle_radius, disk_type='W'):
    """
    디스크 요소 그리기: 큰 원 + 선 + 작은 원
    
    Parameters:
    - disk_type: 'W'(가로선 + 위쪽 작은 원) or 'H'(세로선 + 우측 작은 원)
    """
    validate_disk_type(disk_type)
    
    # 큰 원 그리기
    cv2.circle(img, (rect_center_x, rect_center_y), circle_radius, COLORS['CIRCLE'], 2)
    
    # 선과 작은 원의 위치 계산
    line_half_length = circle_radius
    small_circle_radius = int(SMALL_CIRCLE_SIZE * SCALE / 2)
    
    if disk_type == 'W':
        # W타입: 가로선 + 위쪽 작은 원
        line_start_x = rect_center_x - line_half_length
        line_end_x = rect_center_x + line_half_length
        cv2.line(img, (line_start_x, rect_center_y), (line_end_x, rect_center_y), 
                 COLORS['LINE'], LINE_THICKNESS)
        
        # 원의 상단에 작은 원
        small_circle_pos = (rect_center_x, rect_center_y - circle_radius)
        
    else:  # disk_type == 'H'
        # H타입: 세로선 + 우측 작은 원
        line_start_y = rect_center_y - line_half_length
        line_end_y = rect_center_y + line_half_length
        cv2.line(img, (rect_center_x, line_start_y), (rect_center_x, line_end_y), 
                 COLORS['LINE'], LINE_THICKNESS)
        
        # 원의 우측에 작은 원
        small_circle_pos = (rect_center_x + circle_radius, rect_center_y)
    
    # 작은 원 그리기
    cv2.circle(img, small_circle_pos, small_circle_radius, COLORS['SMALL_CIRCLE'], -1)

def calculate_seat_position(rect_center_x, rect_center_y, circle_radius, disk_type, position_type):
    """seat 위치 계산"""
    if disk_type == 'W':
        # W타입: 가로선 기준
        if position_type == 'A':
            return rect_center_x - circle_radius, rect_center_y  # 시작점 (좌측)
        elif position_type == 'B':
            return rect_center_x, rect_center_y                   # 중심점
        else:  # position_type == 'C'
            return rect_center_x + circle_radius, rect_center_y  # 종점 (우측)
    else:  # disk_type == 'H'
        # H타입: 세로선 기준
        if position_type == 'A':
            return rect_center_x, rect_center_y - circle_radius  # 시작점 (위쪽)
        elif position_type == 'B':
            return rect_center_x, rect_center_y                   # 중심점
        else:  # position_type == 'C'
            return rect_center_x, rect_center_y + circle_radius  # 종점 (아래쪽)

def calculate_triangle_points(center_x, center_y, direction_type):
    """방향에 따른 삼각형 꼭짓점 계산"""
    triangle_size = int(TRIANGLE_SIZE * SCALE)
    triangle_height = int(triangle_size * math.sqrt(3) / 2)
    
    if direction_type == 'F':  # 북쪽 (위)
        pointing_point = (center_x, center_y - triangle_height // 2)
        base_point1 = (center_x - triangle_size // 2, center_y + triangle_height // 2)
        base_point2 = (center_x + triangle_size // 2, center_y + triangle_height // 2)
    elif direction_type == 'R':  # 동쪽 (오른쪽)
        pointing_point = (center_x + triangle_height // 2, center_y)
        base_point1 = (center_x - triangle_height // 2, center_y - triangle_size // 2)
        base_point2 = (center_x - triangle_height // 2, center_y + triangle_size // 2)
    elif direction_type == 'B':  # 남쪽 (아래)
        pointing_point = (center_x, center_y + triangle_height // 2)
        base_point1 = (center_x - triangle_size // 2, center_y - triangle_height // 2)
        base_point2 = (center_x + triangle_size // 2, center_y - triangle_height // 2)
    else:  # direction_type == 'L' (서쪽, 왼쪽)
        pointing_point = (center_x - triangle_height // 2, center_y)
        base_point1 = (center_x + triangle_height // 2, center_y - triangle_size // 2)
        base_point2 = (center_x + triangle_height // 2, center_y + triangle_size // 2)
    
    return pointing_point, base_point1, base_point2

def seat(img, rect_center_x, rect_center_y, circle_radius, disk_type='W', 
         position_type='B', direction_type='F'):
    """
    시트 요소 그리기: 삼각형 + 작은 원
    
    Parameters:
    - disk_type: 'W' or 'H' (위치 계산용)
    - position_type: 'A'(시작점), 'B'(중심), 'C'(종점)
    - direction_type: 'F'(북쪽), 'R'(동쪽), 'B'(남쪽), 'L'(서쪽)
    """
    validate_position_type(position_type)
    validate_direction_type(direction_type)
    
    # 위치 계산
    triangle_center_x, triangle_center_y = calculate_seat_position(
        rect_center_x, rect_center_y, circle_radius, disk_type, position_type)
    
    # 방향에 따른 삼각형 꼭짓점 계산
    pointing_point, base_point1, base_point2 = calculate_triangle_points(
        triangle_center_x, triangle_center_y, direction_type)
    
    # 삼각형 그리기
    triangle_points = np.array([pointing_point, base_point1, base_point2], np.int32)
    triangle_points = triangle_points.reshape((-1, 1, 2))
    cv2.polylines(img, [triangle_points], True, COLORS['TRIANGLE'], 2)
    
    # 방향을 가리키는 꼭짓점에 작은 원 그리기
    small_circle_radius = int(SMALL_CIRCLE_SIZE * SCALE / 2)
    cv2.circle(img, pointing_point, small_circle_radius, COLORS['SMALL_CIRCLE'], -1)

# ===========================================
# 메인 장면 생성 함수
# ===========================================
def create_scene(disk_type='W', position_type='B', direction_type='F'):
    """
    완전한 장면 생성
    
    Parameters:
    - disk_type: 'W'(가로선) or 'H'(세로선)
    - position_type: 'A'(시작점), 'B'(중심), 'C'(종점)
    - direction_type: 'F'(북쪽), 'R'(동쪽), 'B'(남쪽), 'L'(서쪽)
    
    Returns:
    - img: 생성된 이미지 배열
    """
    # 입력값 유효성 검사
    validate_disk_type(disk_type)
    validate_position_type(position_type)
    validate_direction_type(direction_type)
    
    # 캔버스 설정
    canvas_width, canvas_height = get_canvas_setup()
    img = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # 정사각형 설정
    rect_width, rect_height, rect_x, rect_y, rect_center_x, rect_center_y = get_rect_setup(
        canvas_width, canvas_height)
    
    # 원의 반지름
    circle_radius = get_circle_radius()
    
    # 요소들 그리기
    draw_rectangle(img, rect_x, rect_y, rect_width, rect_height)
    disck(img, rect_center_x, rect_center_y, circle_radius, disk_type)
    seat(img, rect_center_x, rect_center_y, circle_radius, disk_type, position_type, direction_type)
    
    return img

def display_and_save_scene(img, disk_type, position_type, direction_type):
    """장면 표시 및 저장"""
    # 파일명 및 창 제목 생성
    filename = f'opencv_{disk_type}_{position_type}_{direction_type}_final.png'
    window_name = f'{disk_type}{position_type}{direction_type} Type - Final Version'
    
    # 이미지 저장 및 표시
    cv2.imwrite(filename, img)
    cv2.imshow(window_name, img)
    
    # 정보 출력
    print_scene_info(disk_type, position_type, direction_type, filename)
    
    # 키 입력 대기
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def print_scene_info(disk_type, position_type, direction_type, filename):
    """장면 정보 출력"""
    # 이름 매핑
    disk_names = {'W': '가로선', 'H': '세로선'}
    position_names = {'A': '시작점', 'B': '중심', 'C': '종점'}
    direction_names = {'F': '북쪽', 'R': '동쪽', 'B': '남쪽', 'L': '서쪽'}
    
    canvas_width, canvas_height = get_canvas_setup()
    rect_width = int(RECT_SIZE * SCALE)
    circle_radius = get_circle_radius()
    
    print("=" * 60)
    print(f"🎨 {disk_type}{position_type}{direction_type}타입 장면이 생성되었습니다!")
    print("=" * 60)
    print(f"📁 파일명: {filename}")
    print(f"📐 캔버스 크기: {canvas_width} x {canvas_height}")
    print(f"⬛ 정사각형: {rect_width} x {rect_width} ({RECT_SIZE}x{RECT_SIZE} 단위)")
    print(f"⭕ 큰 원 반지름: {circle_radius} ({CIRCLE_SIZE}/2 단위)")
    print(f"📏 스케일: {SCALE}")
    print()
    print("🔧 구성 요소:")
    print(f"   • disck: 큰 원 + 작은 원 + {disk_names[disk_type]}")
    print(f"   • seat: 삼각형({direction_names[direction_type]} 방향) + 작은 원")
    print(f"   • 위치: {disk_names[disk_type]}의 {position_names[position_type]}")
    print()
    print("⌨️  키보드의 아무 키나 누르면 창이 닫힙니다.")

# ===========================================
# 메인 실행부
# ===========================================
def main():
    """메인 실행 함수"""
    try:
        # 장면 타입 설정
        disk_type = 'W'        # 'W': 가로선, 'H': 세로선
        position_type = 'B'    # 'A': 시작점, 'B': 중심, 'C': 종점  
        direction_type = 'F'   # 'F': 북쪽, 'R': 동쪽, 'B': 남쪽, 'L': 서쪽
        
        # 장면 생성
        img = create_scene(disk_type, position_type, direction_type)
        
        # 표시 및 저장
        display_and_save_scene(img, disk_type, position_type, direction_type)
        
    except ValueError as e:
        print(f"❌ 입력 오류: {e}")
    except Exception as e:
        print(f"❌ 실행 오류: {e}")

if __name__ == "__main__":
    main()