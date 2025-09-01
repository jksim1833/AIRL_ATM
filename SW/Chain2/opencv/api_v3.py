import cv2, numpy as np, math
from typing import Dict, Tuple, Optional, Literal

DEFAULTS = {
    "SCALE": 80,
    "CANVAS_SIZE": 5,
    "BIG_RECT_SIZE": 8,
    "CIRCLE_SIZE": 3.6,
    "TRIANGLE_SIZE": 1.2,
    "SMALL_CIRCLE_SIZE": 0.24,
    "LINE_THICKNESS": 3,
    "COLORS": {
        "RECT": (0, 0, 0),
        "CIRCLE": (0, 0, 255),
        "LINE": (0, 255, 0),
        "TRIANGLE": (255, 0, 0),
        "SMALL_CIRCLE": (255, 0, 255),
    },
}

def _disk(img: np.ndarray, x: int, y: int, r: int, t: Literal["W","H"],
          colors: Dict[str, Tuple[int,int,int]], s: int, th: int) -> None:
    cv2.circle(img, (x, y), r, colors["CIRCLE"], 2)
    if t == "W":
        cv2.line(img, (x - r, y), (x + r, y), colors["LINE"], th)
        cv2.circle(img, (x, y - r), s, colors["SMALL_CIRCLE"], -1)
    else:
        cv2.line(img, (x, y - r), (x, y + r), colors["LINE"], th)
        cv2.circle(img, (x + r, y), s, colors["SMALL_CIRCLE"], -1)

def _seat(img: np.ndarray, x: int, y: int, r: int,
          t: Literal["W","H"], p: Literal["A","B","C"], d: Literal["F","R","B","L"],
          colors: Dict[str, Tuple[int,int,int]], ts: int, th2: int, s: int) -> None:
    if t == "W":
        lx, ly = (x - r, y) if p == "A" else (x, y) if p == "B" else (x + r, y)
    else:
        lx, ly = (x, y - r) if p == "A" else (x, y) if p == "B" else (x, y + r)
    if d == "F":
        pv, p1, p2 = (lx, ly - (th2*2//3)), (lx - ts//2, ly - (th2*2//3) + th2), (lx + ts//2, ly - (th2*2//3) + th2)
    elif d == "R":
        pv, p1, p2 = (lx + (th2*2//3), ly), (lx + (th2*2//3) - th2, ly - ts//2), (lx + (th2*2//3) - th2, ly + ts//2)
    elif d == "B":
        pv, p1, p2 = (lx, ly + (th2*2//3)), (lx - ts//2, ly + (th2*2//3) - th2), (lx + ts//2, ly + (th2*2//3) - th2)
    else:
        pv, p1, p2 = (lx - (th2*2//3), ly), (lx - (th2*2//3) + th2, ly - ts//2), (lx - (th2*2//3) + th2, ly + ts//2)
    tri = np.array([pv, p1, p2], np.int32).reshape((-1,1,2))
    cv2.polylines(img, [tri], True, colors["TRIANGLE"], 2)
    cv2.circle(img, pv, s, colors["SMALL_CIRCLE"], -1)

def _create_image(cell_configs: Dict[int, Tuple[str,str,str]],
                  scale: int, canvas_size: int, big_rect_size: int,
                  circle_size: float, triangle_size: float, small_circle_size: float,
                  line_thickness: int, colors: Dict[str, Tuple[int,int,int]]) -> Tuple[np.ndarray, Dict]:
    canvas_px = int(canvas_size * scale)
    total = canvas_px * 2
    img = np.ones((total, total, 3), dtype=np.uint8) * 255
    big_px = int(big_rect_size * scale)
    bx = by = (total - big_px) // 2
    cv2.rectangle(img, (bx, by), (bx + big_px, by + big_px), colors["RECT"], 2)
    q = big_px // 4
    centers = {1:(bx+q,by+q), 2:(bx+3*q,by+q), 3:(bx+q,by+3*q), 4:(bx+3*q,by+3*q)}
    r = int(circle_size * scale / 2)
    ts = int(triangle_size * scale)
    th2 = int(ts * math.sqrt(3) / 2)
    s = int(small_circle_size * scale / 2)
    for cid,(dt,pt,dr) in cell_configs.items():
        cx, cy = centers[cid]
        _disk(img, cx, cy, r, dt, colors, s, line_thickness)
        _seat(img, cx, cy, r, dt, pt, dr, colors, ts, th2, s)
    meta = {"image_size": (total,total), "big_rect_top_left": (bx,by), "big_rect_size": big_px,
            "centers": centers, "circle_radius": r, "scale": scale}
    return img, meta

def render_divided_square(
    cell_configs: Dict[int, Tuple[Literal["W","H"], Literal["A","B","C"], Literal["F","R","B","L"]]],
    *, scale: Optional[int]=None, canvas_size: Optional[int]=None, big_rect_size: Optional[int]=None,
    circle_size: Optional[float]=None, triangle_size: Optional[float]=None, small_circle_size: Optional[float]=None,
    line_thickness: Optional[int]=None, colors: Optional[Dict[str, Tuple[int,int,int]]]=None,
    save_path: Optional[str]=None, encode: Optional[Literal[".png",".jpg",".jpeg"]]=None, return_bytes: bool=False
) -> Dict:
    scale = scale or DEFAULTS["SCALE"]
    canvas_size = canvas_size or DEFAULTS["CANVAS_SIZE"]
    big_rect_size = big_rect_size or DEFAULTS["BIG_RECT_SIZE"]
    circle_size = circle_size or DEFAULTS["CIRCLE_SIZE"]
    triangle_size = triangle_size or DEFAULTS["TRIANGLE_SIZE"]
    small_circle_size = small_circle_size or DEFAULTS["SMALL_CIRCLE_SIZE"]
    line_thickness = line_thickness or DEFAULTS["LINE_THICKNESS"]
    colors = colors or DEFAULTS["COLORS"]
    img, meta = _create_image(cell_configs, scale, canvas_size, big_rect_size, circle_size,
                              triangle_size, small_circle_size, line_thickness, colors)
    out: Dict = {"meta": meta, "saved_path": None, "encoded_len": None, "bytes": None}
    if save_path: cv2.imwrite(save_path, img); out["saved_path"] = save_path
    if encode and return_bytes:
        ok, buf = cv2.imencode(encode, img, params=[cv2.IMWRITE_PNG_COMPRESSION, 3] if encode==".png" else [])
        if not ok: raise RuntimeError("Image encoding failed")
        out["bytes"] = buf.tobytes(); out["encoded_len"] = len(out["bytes"])
    return out
