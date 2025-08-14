import cv2
import numpy as np
import math

# ===========================================
# 설정 상수
# ===========================================
SCALE = 40                    # 그리드용 작은 스케일 (1단위 = 40픽셀)
CANVAS_SIZE = 6              # 개별 캔버스 크기 (6 * SCALE)
RECT_SIZE = 2.5              # 정사각형 크기 (2.5 * SCALE)
CIRCLE_SIZE = 2.25           # 큰 원 크기 (2.25 * SCALE)
TRIANGLE_SIZE = 0.75         # 삼각형 크기 (0.75 * SCALE)
SMALL_CIRCLE_SIZE = 0.15     # 작은 원 크기 (0.15 * SCALE)

# 색상 정의 (BGR 형식)
COLORS = {
    'RECT': (0, 0, 0),           # 검은색 - 정사각형
    'CIRCLE': (0, 0, 255),       # 빨간색 - 큰 원
    'LINE': (0, 255, 0),         # 녹색 - 선
    'TRIANGLE': (255, 0, 0),     # 파란색 - 삼각형
    'SMALL_CIRCLE': (255, 0, 255), # 자홍색 - 작은 원
    'BACKGROUND': (255, 255, 255), # 흰색 - 배경
    'GRID_LINE': (200, 200, 200), # 연한 회색 - 그리드 선
    'TEXT': (0, 0, 0)            # 검은색 - 텍스트
}

# 선 두께
LINE_THICKNESS = 1

# 그리드 설정
GRID_ROWS = 4                # 4행 (F, R, B, L)
GRID_COLS = 6                # 6열 (WA, WB, WC, HA, HB, HC)

# ===========================================
# 유틸리티 함수
# ===========================================
def get_canvas_setup():
    """개별 캔버스 기본 설정 반환"""
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
                  COLORS['RECT'], 1)

def disck(img, rect_center_x, rect_center_y, circle_radius, disk_type='W'):
    """디스크 요소 그리기: 큰 원 + 선 + 작은 원"""
    # 큰 원 그리기
    cv2.circle(img, (rect_center_x, rect_center_y), circle_radius, COLORS['CIRCLE'], 1)
    
    # 선과 작은 원의 위치 계산
    line_half_length = circle_radius
    small_circle_radius = int(SMALL_CIRCLE_SIZE * SCALE / 2)
    
    if disk_type == 'W':
        # W타입: 가로선 + 위쪽 작은 원
        line_start_x = rect_center_x - line_half_length
        line_end_x = rect_center_x + line_half_length
        cv2.line(img, (line_start_x, rect_center_y), (line_end_x, rect_center_y), 
                 COLORS['LINE'], LINE_THICKNESS)
        small_circle_pos = (rect_center_x, rect_center_y - circle_radius)
        
    else:  # disk_type == 'H'
        # H타입: 세로선 + 우측 작은 원
        line_start_y = rect_center_y - line_half_length
        line_end_y = rect_center_y + line_half_length
        cv2.line(img, (rect_center_x, line_start_y), (rect_center_x, line_end_y), 
                 COLORS['LINE'], LINE_THICKNESS)
        small_circle_pos = (rect_center_x + circle_radius, rect_center_y)
    
    # 작은 원 그리기
    cv2.circle(img, small_circle_pos, small_circle_radius, COLORS['SMALL_CIRCLE'], -1)

def calculate_seat_position(rect_center_x, rect_center_y, circle_radius, disk_type, position_type):
    """seat 위치 계산"""
    if disk_type == 'W':
        if position_type == 'A':
            return rect_center_x - circle_radius, rect_center_y
        elif position_type == 'B':
            return rect_center_x, rect_center_y
        else:  # position_type == 'C'
            return rect_center_x + circle_radius, rect_center_y
    else:  # disk_type == 'H'
        if position_type == 'A':
            return rect_center_x, rect_center_y - circle_radius
        elif position_type == 'B':
            return rect_center_x, rect_center_y
        else:  # position_type == 'C'
            return rect_center_x, rect_center_y + circle_radius

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
    """시트 요소 그리기: 삼각형 + 작은 원"""
    # 위치 계산
    triangle_center_x, triangle_center_y = calculate_seat_position(
        rect_center_x, rect_center_y, circle_radius, disk_type, position_type)
    
    # 방향에 따른 삼각형 꼭짓점 계산
    pointing_point, base_point1, base_point2 = calculate_triangle_points(
        triangle_center_x, triangle_center_y, direction_type)
    
    # 삼각형 그리기
    triangle_points = np.array([pointing_point, base_point1, base_point2], np.int32)
    triangle_points = triangle_points.reshape((-1, 1, 2))
    cv2.polylines(img, [triangle_points], True, COLORS['TRIANGLE'], 1)
    
    # 방향을 가리키는 꼭짓점에 작은 원 그리기
    small_circle_radius = int(SMALL_CIRCLE_SIZE * SCALE / 2)
    cv2.circle(img, pointing_point, small_circle_radius, COLORS['SMALL_CIRCLE'], -1)

def create_single_scene(disk_type, position_type, direction_type):
    """단일 장면 생성"""
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

# ===========================================
# 그리드 생성 함수
# ===========================================
def create_grid_with_labels():
    """4x6 그리드 생성 (레이블 포함)"""
    # 개별 캔버스 크기
    canvas_width, canvas_height = get_canvas_setup()
    
    # 레이블 영역 크기
    label_height = 30
    label_width = 50
    
    # 전체 그리드 크기 계산
    total_width = GRID_COLS * canvas_width + label_width
    total_height = GRID_ROWS * canvas_height + label_height
    
    # 전체 그리드 이미지 생성
    grid_img = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # 타입 정의
    disk_types = ['W', 'H']
    position_types = ['A', 'B', 'C']
    direction_types = ['F', 'R', 'B', 'L']
    
    # 행 레이블 (방향) - 알파벳만
    direction_labels = ['F', 'R', 'B', 'L']
    
    # 열 레이블 (디스크타입 + 위치)
    col_labels = []
    for disk in disk_types:
        for pos in position_types:
            col_labels.append(f'{disk}{pos}')
    
    # 행 레이블 그리기
    for row, label in enumerate(direction_labels):
        y_pos = label_height + row * canvas_height + canvas_height // 2
        cv2.putText(grid_img, label, (5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['TEXT'], 1)
    
    # 열 레이블 그리기
    for col, label in enumerate(col_labels):
        x_pos = label_width + col * canvas_width + canvas_width // 2 - 15
        cv2.putText(grid_img, label, (x_pos, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['TEXT'], 1)
    
    # 각 셀에 장면 그리기
    for row, direction in enumerate(direction_types):
        for col, (disk, position) in enumerate([(d, p) for d in disk_types for p in position_types]):
            # 단일 장면 생성
            scene_img = create_single_scene(disk, position, direction)
            
            # 그리드에서의 위치 계산
            start_x = label_width + col * canvas_width
            end_x = start_x + canvas_width
            start_y = label_height + row * canvas_height
            end_y = start_y + canvas_height
            
            # 장면을 그리드에 복사
            grid_img[start_y:end_y, start_x:end_x] = scene_img
    
    # 그리드 선 그리기
    # 세로선
    for col in range(GRID_COLS + 1):
        x = label_width + col * canvas_width
        cv2.line(grid_img, (x, label_height), (x, total_height), COLORS['GRID_LINE'], 1)
    
    # 가로선
    for row in range(GRID_ROWS + 1):
        y = label_height + row * canvas_height
        cv2.line(grid_img, (label_width, y), (total_width, y), COLORS['GRID_LINE'], 1)
    
    # 레이블 구분선
    cv2.line(grid_img, (0, label_height), (total_width, label_height), COLORS['TEXT'], 2)
    cv2.line(grid_img, (label_width, 0), (label_width, total_height), COLORS['TEXT'], 2)
    
    return grid_img

def print_grid_info():
    """그리드 정보 출력"""
    canvas_width, canvas_height = get_canvas_setup()
    
    print("=" * 80)
    print("🎨 OpenCV 4x6 그리드 결과표 생성 완료!")
    print("=" * 80)
    print(f"📊 그리드 구성: {GRID_ROWS}행 × {GRID_COLS}열 = 24개 조합")
    print(f"📐 개별 캔버스: {canvas_width} × {canvas_height} 픽셀")
    print(f"📏 스케일: {SCALE} (그리드용 축소)")
    print()
    print("🔤 행 (세로): 방향값")
    print("   • F: 북쪽 (위)")
    print("   • R: 동쪽 (오른쪽)")
    print("   • B: 남쪽 (아래)")
    print("   • L: 서쪽 (왼쪽)")
    print()
    print("🔤 열 (가로): 디스크타입 + 위치값")
    print("   • W: 가로선, H: 세로선")
    print("   • A: 시작점, B: 중심, C: 종점")
    print("   • 조합: WA, WB, WC, HA, HB, HC")
    print()
    print("⌨️  키보드의 아무 키나 누르면 창이 닫힙니다.")

# ===========================================
# 메인 실행부
# ===========================================
def main():
    """메인 실행 함수"""
    try:
        print("🔄 4x6 그리드 생성 중...")
        
        # 그리드 생성
        grid_img = create_grid_with_labels()
        
        # 파일 저장
        filename = 'opencv_4x6_grid_results.png'
        cv2.imwrite(filename, grid_img)
        
        # 화면 표시
        cv2.imshow('OpenCV 4x6 Grid Results', grid_img)
        
        # 정보 출력
        print_grid_info()
        print(f"📁 파일 저장: {filename}")
        
        # 키 입력 대기
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("✅ 그리드 생성 완료!")
        
    except Exception as e:
        print(f"❌ 실행 오류: {e}")

if __name__ == "__main__":
    main()