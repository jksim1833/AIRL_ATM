import cv2
import numpy as np
import math

# ===========================================
# 설정 상수
# ===========================================
SCALE = 80                    # 해상도 스케일 (1단위 = 80픽셀)
CANVAS_SIZE = 10             # 개별 셀 캔버스 크기 (10 * SCALE)
RECT_SIZE = 4                # 개별 셀의 정사각형 크기 (4 * SCALE)
BIG_RECT_SIZE = 8            # 큰 정사각형 크기 (개별 셀의 2배)
CIRCLE_SIZE = 3.6            # 큰 원 크기 (3.6 * SCALE)
TRIANGLE_SIZE = 1.2          # 삼각형 크기 (1.2 * SCALE)
SMALL_CIRCLE_SIZE = 0.24     # 작은 원 크기 (0.24 * SCALE)

# 색상 정의 (BGR 형식)
COLORS = {
    'RECT': (0, 0, 0),           # 검은색 - 정사각형
    'CIRCLE': (0, 0, 255),       # 빨간색 - 큰 원
    'LINE': (0, 255, 0),         # 녹색 - 선
    'TRIANGLE': (255, 0, 0),     # 파란색 - 삼각형
    'SMALL_CIRCLE': (255, 0, 255), # 자홍색 - 작은 원
    'BACKGROUND': (255, 255, 255), # 흰색 - 배경
    'GRID_LINE': (128, 128, 128)   # 회색 - 그리드 선
}

# 선 두께
LINE_THICKNESS = 3

# 그리드 설정
GRID_ROWS = 2                # 2행
GRID_COLS = 2                # 2열

# ===========================================
# 유틸리티 함수
# ===========================================
def validate_cell_id(cell_id):
    """cell_id 유효성 검사"""
    if cell_id not in (1, 2, 3, 4):
        raise ValueError("cell_id must be 1, 2, 3, or 4")

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

def get_cell_canvas_setup():
    """개별 셀 캔버스 기본 설정 반환"""
    canvas_width = int(CANVAS_SIZE * SCALE)
    canvas_height = int(CANVAS_SIZE * SCALE)
    return canvas_width, canvas_height

def get_cell_position(cell_id, cell_canvas_width, cell_canvas_height):
    """cell_id에 따른 그리드에서의 위치 계산"""
    validate_cell_id(cell_id)
    
    # cell_id에 따른 행, 열 계산
    # 1: 좌측 상단, 2: 우측 상단, 3: 좌측 하단, 4: 우측 하단
    if cell_id == 1:
        row, col = 0, 0  # 좌측 상단
    elif cell_id == 2:
        row, col = 0, 1  # 우측 상단
    elif cell_id == 3:
        row, col = 1, 0  # 좌측 하단
    else:  # cell_id == 4
        row, col = 1, 1  # 우측 하단
    
    # 그리드에서의 실제 위치 계산
    start_x = col * cell_canvas_width
    start_y = row * cell_canvas_height
    
    return start_x, start_y, row, col

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
# 그리기 함수 (개별 셀용)
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
# 개별 셀 생성 함수
# ===========================================
def create_single_cell(disk_type='W', position_type='B', direction_type='F', draw_background=False):
    """
    개별 셀 생성 (기본 코드와 동일한 구조)
    
    Parameters:
    - disk_type: 'W'(가로선) or 'H'(세로선)
    - position_type: 'A'(시작점), 'B'(중심), 'C'(종점)
    - direction_type: 'F'(북쪽), 'R'(동쪽), 'B'(남쪽), 'L'(서쪽)
    - draw_background: 배경을 그릴지 여부 (통합시에는 False)
    
    Returns:
    - img: 생성된 셀 이미지 배열
    """
    # 입력값 유효성 검사
    validate_disk_type(disk_type)
    validate_position_type(position_type)
    validate_direction_type(direction_type)
    
    # 캔버스 설정
    canvas_width, canvas_height = get_cell_canvas_setup()
    
    if draw_background:
        img = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    else:
        # 투명한 배경 (실제로는 사용하지 않고 통합 이미지에 직접 그림)
        img = None
    
    # 정사각형 설정
    rect_width, rect_height, rect_x, rect_y, rect_center_x, rect_center_y = get_rect_setup(
        canvas_width, canvas_height)
    
    # 원의 반지름
    circle_radius = get_circle_radius()
    
    return rect_width, rect_height, rect_x, rect_y, rect_center_x, rect_center_y, circle_radius

# ===========================================
# 2x2 그리드 생성 함수
# ===========================================
def create_2x2_grid(cell_configs):
    """
    하나의 큰 정사각형을 4분할해서 4개 셀 배치
    
    Parameters:
    - cell_configs: 딕셔너리 {cell_id: (disk_type, position_type, direction_type)}
                   예: {1: ('W', 'B', 'F'), 2: ('H', 'A', 'R'), ...}
    
    Returns:
    - grid_img: 통합된 정사각형 이미지
    """
    # 개별 셀 캔버스 크기
    cell_canvas_width, cell_canvas_height = get_cell_canvas_setup()
    
    # 전체 정사각형 크기 계산
    total_width = GRID_COLS * cell_canvas_width
    total_height = GRID_ROWS * cell_canvas_height
    
    # 전체 정사각형 이미지 생성 (흰색 배경)
    grid_img = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # 큰 정사각형의 크기와 위치 계산 (전체 영역의 중앙에 배치)
    big_rect_size = int(BIG_RECT_SIZE * SCALE)
    big_rect_x = (total_width - big_rect_size) // 2
    big_rect_y = (total_height - big_rect_size) // 2
    
    # 하나의 큰 정사각형 그리기 (2배 크기)
    cv2.rectangle(grid_img, (big_rect_x, big_rect_y), 
                  (big_rect_x + big_rect_size, big_rect_y + big_rect_size), 
                  COLORS['RECT'], 2)
    
    # 4분할 선 그리기
    # 세로 중앙선
    center_x = big_rect_x + big_rect_size // 2
    cv2.line(grid_img, (center_x, big_rect_y), (center_x, big_rect_y + big_rect_size), 
             COLORS['RECT'], 2)
    
    # 가로 중앙선  
    center_y = big_rect_y + big_rect_size // 2
    cv2.line(grid_img, (big_rect_x, center_y), (big_rect_x + big_rect_size, center_y), 
             COLORS['RECT'], 2)
    
    # 각 셀의 중심점 계산 (4분할된 각 영역의 중심)
    quarter_size = big_rect_size // 2
    cell_centers = {
        1: (big_rect_x + quarter_size // 2, big_rect_y + quarter_size // 2),      # 좌상단
        2: (big_rect_x + quarter_size + quarter_size // 2, big_rect_y + quarter_size // 2), # 우상단
        3: (big_rect_x + quarter_size // 2, big_rect_y + quarter_size + quarter_size // 2), # 좌하단
        4: (big_rect_x + quarter_size + quarter_size // 2, big_rect_y + quarter_size + quarter_size // 2) # 우하단
    }
    
    # 각 셀의 내용물 그리기 (4분할된 영역에 맞춰서)
    for cell_id in range(1, 5):
        if cell_id in cell_configs:
            disk_type, position_type, direction_type = cell_configs[cell_id]
            
            # 해당 셀의 중심점
            cell_center_x, cell_center_y = cell_centers[cell_id]
            
            # 원의 반지름 (4분할 영역에 맞게 조정)
            circle_radius = int(CIRCLE_SIZE * SCALE / 2)
            
            # 디스크와 시트 요소 그리기
            disck(grid_img, cell_center_x, cell_center_y, circle_radius, disk_type)
            seat(grid_img, cell_center_x, cell_center_y, circle_radius, disk_type, position_type, direction_type)
    
    return grid_img

def add_cell_labels(grid_img):
    """셀에 ID 레이블 추가 (선택적)"""
    cell_canvas_width, cell_canvas_height = get_cell_canvas_setup()
    
    for cell_id in range(1, 5):
        start_x, start_y, row, col = get_cell_position(cell_id, cell_canvas_width, cell_canvas_height)
        
        # 레이블 위치 (셀의 좌측 상단 모서리, 작게 표시)
        label_x = start_x + 15
        label_y = start_y + 25
        
        # 레이블 그리기 (작고 연하게)
        cv2.putText(grid_img, f'{cell_id}', (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# ===========================================
# 메인 실행부
# ===========================================
def main():
    """메인 실행 함수"""
    try:
        # 각 셀의 설정 (cell_id: (disk_type, position_type, direction_type))
        cell_configs = {
            1: ('W', 'B', 'F'),  # 셀 1: 가로선, 중심, 북쪽
            2: ('H', 'A', 'R'),  # 셀 2: 세로선, 시작점, 동쪽
            3: ('W', 'C', 'B'),  # 셀 3: 가로선, 종점, 남쪽
            4: ('H', 'B', 'L')   # 셀 4: 세로선, 중심, 서쪽
        }
        
        print("🔄 4분할 정사각형 (4개 셀) 생성 중...")
        
        # 4분할 정사각형 생성
        square_img = create_2x2_grid(cell_configs)
        
        # 파일 저장
        filename = 'opencv_divided_square.png'
        cv2.imwrite(filename, square_img)
        
        # 화면 표시
        cv2.imshow('OpenCV Divided Square - 4 Cells', square_img)
        
        # 정보 출력
        cell_canvas_width, cell_canvas_height = get_cell_canvas_setup()
        total_width = GRID_COLS * cell_canvas_width
        total_height = GRID_ROWS * cell_canvas_height
        big_rect_size = int(BIG_RECT_SIZE * SCALE)
        
        print("=" * 60)
        print("🎨 4분할 정사각형 (4개 셀) 생성 완료!")
        print("=" * 60)
        print(f"📁 파일명: {filename}")
        print(f"📐 전체 캔버스: {total_width} x {total_height}")
        print(f"🔳 큰 정사각형: {big_rect_size} x {big_rect_size} (개별 셀의 2배)")
        print(f"📏 개별 셀 영역: {big_rect_size//2} x {big_rect_size//2}")
        print(f"🔀 구조: 하나의 큰 정사각형을 4분할")
        print()
        print("🔢 셀 구성 (4분할 배치):")
        cell_positions = {1: "좌상단", 2: "우상단", 3: "좌하단", 4: "우하단"}
        for cell_id, (disk, pos, direction) in cell_configs.items():
            position_names = {'A': '시작점', 'B': '중심', 'C': '종점'}
            disk_names = {'W': '가로선', 'H': '세로선'}
            direction_names = {'F': '북쪽', 'R': '동쪽', 'B': '남쪽', 'L': '서쪽'}
            print(f"   셀 {cell_id} ({cell_positions[cell_id]}): {disk_names[disk]}, {position_names[pos]}, {direction_names[direction]} ({disk}{pos}{direction})")
        print()
        print("⌨️  키보드의 아무 키나 누르면 창이 닫힙니다.")
        
        # 키 입력 대기
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("✅ 4분할 정사각형 생성 완료!")
        
    except Exception as e:
        print(f"❌ 실행 오류: {e}")

if __name__ == "__main__":
    main()