# diagram_api 테스트 및 출력 확인 코드
from api_v2 import render_divided_square
import cv2

# 테스트 설정
test_config = {
    1: ("W", "A", "F"),  # Seat 1 with disk type "W", left position "A", facing forward "F"
    2: ("H", "C", "F"),  # Seat 2 with disk type "W", right position "C", facing forward "F"
    3: ("W", "A", "F"),  # Seat 3 with disk type "H", left position "A", facing forward "F"
    4: ("H", "C", "F")  # 셀4: W타입 디스크, C위치, 좌측 방향
}

# 다이어그램 생성 및 파일로 저장
result = render_divided_square(
    cell_configs=test_config,
    save_path="test_output.png"  # 현재 디렉토리에 저장
)

print("이미지 생성 완료!")
print(f"저장된 파일: {result['saved_path']}")
print(f"이미지 크기: {result['meta']['image_size']}")
print(f"사각형 위치: {result['meta']['big_rect_top_left']}")
print(f"사각형 크기: {result['meta']['big_rect_size']}")
print(f"셀 중심점들: {result['meta']['centers']}")