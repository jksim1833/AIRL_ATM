import cv2
import numpy as np
import math

# ===========================================
# 설정 상수
# ===========================================
SCALE = 80
CANVAS_SIZE = 5
BIG_RECT_SIZE = 8
CIRCLE_SIZE = 3.6
TRIANGLE_SIZE = 1.2
SMALL_CIRCLE_SIZE = 0.24
LINE_THICKNESS = 3

# 색상 정의 (BGR)
COLORS = {
    'RECT': (0, 0, 0),
    'CIRCLE': (0, 0, 255),
    'LINE': (0, 255, 0),
    'TRIANGLE': (255, 0, 0),
    'SMALL_CIRCLE': (255, 0, 255)
}

# ===========================================
# 그리기 함수
# ===========================================
def disk(img, center_x, center_y, radius, disk_type):
    """디스크 요소: 큰 원 + 선 + 작은 원"""
    # 큰 원
    cv2.circle(img, (center_x, center_y), radius, COLORS['CIRCLE'], 2)
    
    # 선과 작은 원
    small_radius = int(SMALL_CIRCLE_SIZE * SCALE / 2)
    
    if disk_type == 'W':
        # 가로선 + 위쪽 작은 원
        cv2.line(img, (center_x - radius, center_y), (center_x + radius, center_y), 
                 COLORS['LINE'], LINE_THICKNESS)
        cv2.circle(img, (center_x, center_y - radius), small_radius, COLORS['SMALL_CIRCLE'], -1)
    else:  # H
        # 세로선 + 우측 작은 원
        cv2.line(img, (center_x, center_y - radius), (center_x, center_y + radius), 
                 COLORS['LINE'], LINE_THICKNESS)
        cv2.circle(img, (center_x + radius, center_y), small_radius, COLORS['SMALL_CIRCLE'], -1)

def seat(img, center_x, center_y, radius, disk_type, position_type, direction_type):
    """시트 요소: 삼각형 + 작은 원 (선과의 교차점이 꼭짓점에서 2/3 지점)"""
    # 위치 계산 (선의 위치)
    if disk_type == 'W':
        if position_type == 'A':
            line_x, line_y = center_x - radius, center_y
        elif position_type == 'B':
            line_x, line_y = center_x, center_y
        else:  # C
            line_x, line_y = center_x + radius, center_y
    else:  # H
        if position_type == 'A':
            line_x, line_y = center_x, center_y - radius
        elif position_type == 'B':
            line_x, line_y = center_x, center_y
        else:  # C
            line_x, line_y = center_x, center_y + radius
    
    # 삼각형 크기
    size = int(TRIANGLE_SIZE * SCALE)
    height = int(size * math.sqrt(3) / 2)
    
    # 방향별 삼각형 위치 계산 (꼭짓점에서 2/3 지점이 선과 만나도록)
    if direction_type == 'F':  # 북쪽
        # 북쪽 꼭짓점에서 2/3 지점이 선과 만나도록
        pointing_vertex_y = line_y - (height * 2 // 3)
        pointing_vertex = (line_x, pointing_vertex_y)
        base_point1 = (line_x - size//2, pointing_vertex_y + height)
        base_point2 = (line_x + size//2, pointing_vertex_y + height)
        
    elif direction_type == 'R':  # 동쪽
        # 동쪽 꼭짓점에서 2/3 지점이 선과 만나도록
        pointing_vertex_x = line_x + (height * 2 // 3)
        pointing_vertex = (pointing_vertex_x, line_y)
        base_point1 = (pointing_vertex_x - height, line_y - size//2)
        base_point2 = (pointing_vertex_x - height, line_y + size//2)
        
    elif direction_type == 'B':  # 남쪽
        # 남쪽 꼭짓점에서 2/3 지점이 선과 만나도록
        pointing_vertex_y = line_y + (height * 2 // 3)
        pointing_vertex = (line_x, pointing_vertex_y)
        base_point1 = (line_x - size//2, pointing_vertex_y - height)
        base_point2 = (line_x + size//2, pointing_vertex_y - height)
        
    else:  # L 서쪽
        # 서쪽 꼭짓점에서 2/3 지점이 선과 만나도록
        pointing_vertex_x = line_x - (height * 2 // 3)
        pointing_vertex = (pointing_vertex_x, line_y)
        base_point1 = (pointing_vertex_x + height, line_y - size//2)
        base_point2 = (pointing_vertex_x + height, line_y + size//2)
    
    # 삼각형 그리기
    points = [pointing_vertex, base_point1, base_point2]
    triangle_points = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [triangle_points], True, COLORS['TRIANGLE'], 2)
    
    # 방향 꼭짓점에 작은 원
    small_radius = int(SMALL_CIRCLE_SIZE * SCALE / 2)
    cv2.circle(img, pointing_vertex, small_radius, COLORS['SMALL_CIRCLE'], -1)

# ===========================================
# 메인 생성 함수
# ===========================================
def create_divided_square(cell_configs):
    """4분할 정사각형 생성"""
    # 캔버스 설정
    canvas_size = int(CANVAS_SIZE * SCALE)
    total_size = canvas_size * 2
    img = np.ones((total_size, total_size, 3), dtype=np.uint8) * 255
    
    # 큰 정사각형 설정
    big_rect_size = int(BIG_RECT_SIZE * SCALE)
    big_rect_x = (total_size - big_rect_size) // 2
    big_rect_y = (total_size - big_rect_size) // 2
    
    # 큰 정사각형 그리기
    cv2.rectangle(img, (big_rect_x, big_rect_y), 
                  (big_rect_x + big_rect_size, big_rect_y + big_rect_size), 
                  COLORS['RECT'], 2)
    
    # 4분할 선
    center_x = big_rect_x + big_rect_size // 2
    center_y = big_rect_y + big_rect_size // 2
    cv2.line(img, (center_x, big_rect_y), (center_x, big_rect_y + big_rect_size), COLORS['RECT'], 2)
    cv2.line(img, (big_rect_x, center_y), (big_rect_x + big_rect_size, center_y), COLORS['RECT'], 2)
    
    # 각 셀 중심점
    quarter = big_rect_size // 4
    cell_centers = {
        1: (big_rect_x + quarter, big_rect_y + quarter),
        2: (big_rect_x + 3*quarter, big_rect_y + quarter),
        3: (big_rect_x + quarter, big_rect_y + 3*quarter),
        4: (big_rect_x + 3*quarter, big_rect_y + 3*quarter)
    }
    
    # 각 셀 그리기
    circle_radius = int(CIRCLE_SIZE * SCALE / 2)
    for cell_id, (disk_type, position_type, direction_type) in cell_configs.items():
        cx, cy = cell_centers[cell_id]
        disk(img, cx, cy, circle_radius, disk_type)
        seat(img, cx, cy, circle_radius, disk_type, position_type, direction_type)
    
    return img

# ===========================================
# 메인 실행
# ===========================================
def main():
    # 셀 설정
    cell_configs = {
        1: ('W', 'B', 'F'),  # 가로선, 중심, 북쪽
        2: ('H', 'A', 'R'),  # 세로선, 시작점, 동쪽
        3: ('W', 'C', 'B'),  # 가로선, 종점, 남쪽
        4: ('H', 'B', 'L')   # 세로선, 중심, 서쪽
    }
    
    # 이미지 생성
    img = create_divided_square(cell_configs)
    
    # 저장 및 표시
    cv2.imwrite('opencv_divided_square.png', img)
    cv2.imshow('4-Divided Square', img)
    
    print("✅ 4분할 정사각형 생성 완료!")
    print("📁 파일: opencv_divided_square.png")
    print("⌨️  아무 키나 누르면 종료")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()