# diagram_api.py
import cv2
import numpy as np
import math
from typing import Dict, Tuple, Optional, Literal

# =========================
# 기본 상수 (필요 시 파라미터로 덮어쓰기 가능)
# =========================
DEFAULTS = {
    "SCALE": 80,
    "CANVAS_SIZE": 5,
    "BIG_RECT_SIZE": 8,
    "CIRCLE_SIZE": 3.6,
    "TRIANGLE_SIZE": 1.2,
    "SMALL_CIRCLE_SIZE": 0.24,
    "LINE_THICKNESS": 3,
    # BGR
    "COLORS": {
        "RECT": (0, 0, 0),
        "CIRCLE": (0, 0, 255),
        "LINE": (0, 255, 0),
        "TRIANGLE": (255, 0, 0),
        "SMALL_CIRCLE": (255, 0, 255),
    },
}

def _disk(img: np.ndarray, center_x: int, center_y: int, radius: int, disk_type: Literal["W", "H"],
          colors: Dict[str, Tuple[int, int, int]], small_r: int, line_thickness: int) -> None:
    cv2.circle(img, (center_x, center_y), radius, colors["CIRCLE"], 2)
    if disk_type == "W":
        cv2.line(img, (center_x - radius, center_y), (center_x + radius, center_y), colors["LINE"], line_thickness)
        cv2.circle(img, (center_x, center_y - radius), small_r, colors["SMALL_CIRCLE"], -1)
    else:
        cv2.line(img, (center_x, center_y - radius), (center_x, center_y + radius), colors["LINE"], line_thickness)
        cv2.circle(img, (center_x + radius, center_y), small_r, colors["SMALL_CIRCLE"], -1)

def _seat(img: np.ndarray, center_x: int, center_y: int, radius: int,
          disk_type: Literal["W", "H"], position_type: Literal["A", "B", "C"],
          direction_type: Literal["F", "R", "B", "L"],
          colors: Dict[str, Tuple[int, int, int]], tri_size: int, tri_h: int, small_r: int) -> None:
    # 선 위치
    if disk_type == "W":
        if position_type == "A":
            line_x, line_y = center_x - radius, center_y
        elif position_type == "B":
            line_x, line_y = center_x, center_y
        else:  # "C"
            line_x, line_y = center_x + radius, center_y
    else:
        if position_type == "A":
            line_x, line_y = center_x, center_y - radius
        elif position_type == "B":
            line_x, line_y = center_x, center_y
        else:
            line_x, line_y = center_x, center_y + radius

    # 꼭짓점(방향)에서 2/3 지점이 선과 만남
    if direction_type == "F":  # 북
        pv_y = line_y - (tri_h * 2 // 3)
        pv = (line_x, pv_y)
        p1 = (line_x - tri_size // 2, pv_y + tri_h)
        p2 = (line_x + tri_size // 2, pv_y + tri_h)
    elif direction_type == "R":  # 동
        pv_x = line_x + (tri_h * 2 // 3)
        pv = (pv_x, line_y)
        p1 = (pv_x - tri_h, line_y - tri_size // 2)
        p2 = (pv_x - tri_h, line_y + tri_size // 2)
    elif direction_type == "B":  # 남
        pv_y = line_y + (tri_h * 2 // 3)
        pv = (line_x, pv_y)
        p1 = (line_x - tri_size // 2, pv_y - tri_h)
        p2 = (line_x + tri_size // 2, pv_y - tri_h)
    else:  # "L" 서
        pv_x = line_x - (tri_h * 2 // 3)
        pv = (pv_x, line_y)
        p1 = (pv_x + tri_h, line_y - tri_size // 2)
        p2 = (pv_x + tri_h, line_y + tri_size // 2)

    tri = np.array([pv, p1, p2], np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [tri], True, colors["TRIANGLE"], 2)
    cv2.circle(img, pv, small_r, colors["SMALL_CIRCLE"], -1)

def _create_image(cell_configs: Dict[int, Tuple[str, str, str]],
                  scale: int, canvas_size: int, big_rect_size: int,
                  circle_size: float, triangle_size: float, small_circle_size: float,
                  line_thickness: int, colors: Dict[str, Tuple[int, int, int]]) -> Tuple[np.ndarray, Dict]:
    """실제 이미지를 생성하고, 메타데이터(중심점 등)를 함께 반환."""
    canvas_px = int(canvas_size * scale)
    total = canvas_px * 2
    img = np.ones((total, total, 3), dtype=np.uint8) * 255

    big_px = int(big_rect_size * scale)
    bx = (total - big_px) // 2
    by = (total - big_px) // 2

    # 외곽 + 십자 분할
    cv2.rectangle(img, (bx, by), (bx + big_px, by + big_px), colors["RECT"], 2)
    cx = bx + big_px // 2
    cy = by + big_px // 2
    cv2.line(img, (cx, by), (cx, by + big_px), colors["RECT"], 2)
    cv2.line(img, (bx, cy), (bx + big_px, cy), colors["RECT"], 2)

    # 셀 중심
    q = big_px // 4
    centers = {
        1: (bx + q, by + q),
        2: (bx + 3 * q, by + q),
        3: (bx + q, by + 3 * q),
        4: (bx + 3 * q, by + 3 * q),
    }

    r = int(circle_size * scale / 2)
    tri_size = int(triangle_size * scale)
    tri_h = int(tri_size * math.sqrt(3) / 2)
    small_r = int(small_circle_size * scale / 2)

    for cell_id, (disk_t, pos_t, dir_t) in cell_configs.items():
        x, y = centers[cell_id]
        _disk(img, x, y, r, disk_t, colors, small_r, line_thickness)
        _seat(img, x, y, r, disk_t, pos_t, dir_t, colors, tri_size, tri_h, small_r)

    meta = {
        "image_size": (total, total),
        "big_rect_top_left": (bx, by),
        "big_rect_size": big_px,
        "centers": centers,
        "circle_radius": r,
        "scale": scale,
    }
    return img, meta

def render_divided_square(
    cell_configs: Dict[int, Tuple[Literal["W","H"], Literal["A","B","C"], Literal["F","R","B","L"]]],
    *,
    # 파라미터 오버라이드 (미지정 시 DEFAULTS 사용)
    scale: Optional[int] = None,
    canvas_size: Optional[int] = None,
    big_rect_size: Optional[int] = None,
    circle_size: Optional[float] = None,
    triangle_size: Optional[float] = None,
    small_circle_size: Optional[float] = None,
    line_thickness: Optional[int] = None,
    colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    # I/O 옵션
    save_path: Optional[str] = None,                      # 파일 저장 (옵션)
    encode: Optional[Literal[".png", ".jpg", ".jpeg"]] = None,  # 바이트 인코딩 (옵션)
    return_bytes: bool = False                            # True면 인코딩 바이트 반환
) -> Dict:
    
    # 파라미터 확정(오버라이드 → 기본값)
    scale = scale or DEFAULTS["SCALE"]
    canvas_size = canvas_size or DEFAULTS["CANVAS_SIZE"]
    big_rect_size = big_rect_size or DEFAULTS["BIG_RECT_SIZE"]
    circle_size = circle_size or DEFAULTS["CIRCLE_SIZE"]
    triangle_size = triangle_size or DEFAULTS["TRIANGLE_SIZE"]
    small_circle_size = small_circle_size or DEFAULTS["SMALL_CIRCLE_SIZE"]
    line_thickness = line_thickness or DEFAULTS["LINE_THICKNESS"]
    colors = colors or DEFAULTS["COLORS"]

    # 이미지 생성(내부는 BGR 유지, 변환 없음)
    img, meta = _create_image(
        cell_configs=cell_configs,
        scale=scale,
        canvas_size=canvas_size,
        big_rect_size=big_rect_size,
        circle_size=circle_size,
        triangle_size=triangle_size,
        small_circle_size=small_circle_size,
        line_thickness=line_thickness,
        colors=colors,
    )

    out: Dict = {"meta": meta, "saved_path": None, "encoded_len": None, "bytes": None}

    # 필요할 때만 파일 저장
    if save_path:
        cv2.imwrite(save_path, img)
        out["saved_path"] = save_path

    # 필요할 때만 바이트 인코딩
    if encode and return_bytes:
        # OpenCV는 BGR을 그대로 인코딩하므로 추가 변환 없음 (PNG/JPG 뷰어에서 정상 표시)
        ok, buf = cv2.imencode(encode, img, params=[cv2.IMWRITE_PNG_COMPRESSION, 3] if encode == ".png" else [])
        if not ok:
            raise RuntimeError("Image encoding failed")
        out["bytes"] = buf.tobytes()
        out["encoded_len"] = len(out["bytes"])

    return out


if __name__ == "__main__":
    cfg = {
        1: ("W", "B", "F"),
        2: ("H", "A", "R"),
        3: ("W", "C", "B"),
        4: ("H", "B", "L"),
    }

    result = render_divided_square(cfg)
    print(result["meta"])
