# chain3_set.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# ===================== 기본 경로 =====================
DEFAULT_BASE = Path.home() / "Desktop" / "AIRL_ATM" / "SW" / "tetris" / "answer_sheet" / "chain3"
DEFAULT_CHAIN2_BASE = Path.home() / "Desktop" / "AIRL_ATM" / "SW" / "tetris" / "answer_sheet" / "chain2"

# ===================== 시나리오 목록 & 기본 매핑 (chain2와 동일해야 함) =====================
SCENARIOS: List[str] = [
    "0", "1-1", "1-2", "1-3", "1-4", "1-5", "2", "3", "4", "5",
    "6-1", "6-2", "7", "8-1", "8-2", "9", "10-1", "10-2",
    "11-1", "11-2", "12", "13", "14", "15", "16", "17-1", "17-2", "18", "19"
]
SCENARIO_SIZE_MAP_DEFAULT: Dict[str, List[str]] = {
    "0": ["L"],
    "1-1": ["L"], "1-2": ["L"], "1-3": ["L"], "1-4": ["L"],
    "1-5": ["M"],
    "2": ["L"], "3": ["L"],
    "4": ["S"],
    "5": ["L"],
    "6-1": ["L"], "6-2": ["L"],
    "7": ["S"],
    "8-1": ["L"], "8-2": ["L"],
    "9": ["S"],
    "10-1": ["M", "L"],
    "10-2": ["M", "L"],
    "11-1": ["S", "M"],
    "11-2": ["S", "M"],
    "12": [],  # 보류
    "13": ["L"],
    "14": ["M"],
    "15": ["S"],
    "16": ["M", "L"],
    "17-1": ["M", "S"],
    "17-2": ["M", "S"],
    "18": ["L"],
    "19": ["L"],
}
PEOPLE_COUNTS: List[int] = [0, 1, 2, 3, 4]

# ===================== chain2 옵션표 (index-사이즈 정렬 자동화에 사용) =====================
OPTION_LIST = {
    "option_list": [
        {
            "people_count": 0,
            "cases": [
                {"luggage_amount": "S", "instruction": {"seats": {
                    "1": ["x", "A", "R", "storage"], "2": ["x", "C", "L", "storage"],
                    "3": ["x", "A", "R", "storage"], "4": ["x", "C", "L", "storage"]}}},
                {"luggage_amount": "M", "instruction": {"seats": {
                    "1": ["x", "A", "R", "storage"], "2": ["x", "C", "L", "storage"],
                    "3": ["x", "A", "R", "storage"], "4": ["x", "C", "L", "storage"]}}},
                {"luggage_amount": "L", "instruction": {"seats": {
                    "1": ["x", "A", "R", "storage"], "2": ["x", "C", "L", "storage"],
                    "3": ["x", "A", "R", "storage"], "4": ["x", "C", "L", "storage"]}}},
            ],
        },
        {
            "people_count": 1,
            "cases": [
                {"luggage_amount": "S", "instruction": {"seats": {
                    "1": ["y", "C", "F", "chair"], "2": ["x", "C", "L", "storage"],
                    "3": ["x", "A", "R", "storage"], "4": ["x", "C", "L", "storage"]}}},
                {"luggage_amount": "M", "instruction": {"seats": {
                    "1": ["x", "M", "F", "chair"], "2": ["x", "C", "L", "storage"],
                    "3": ["x", "A", "R", "storage"], "4": ["x", "C", "L", "storage"]}}},
                {"luggage_amount": "L", "instruction": {"seats": {
                    "1": ["y", "A", "B", "chair"], "2": ["x", "C", "L", "storage"],
                    "3": ["x", "A", "R", "storage"], "4": ["x", "C", "L", "storage"]}}},
            ],
        },
        {
            "people_count": 2,
            "cases": [
                {"luggage_amount": "S", "instruction": {"seats": {
                    "1": ["y", "C", "F", "chair"], "2": ["y", "C", "F", "chair"],
                    "3": ["x", "A", "R", "storage"], "4": ["x", "C", "L", "storage"]}}},
                {"luggage_amount": "M", "instruction": {"seats": {
                    "1": ["x", "M", "F", "chair"], "2": ["x", "M", "F", "chair"],
                    "3": ["x", "A", "R", "storage"], "4": ["x", "C", "L", "storage"]}}},
                {"luggage_amount": "L", "instruction": {"seats": {
                    "1": ["x", "A", "R", "storage"], "2": ["x", "C", "L", "chair"],
                    "3": ["x", "A", "R", "storage"], "4": ["x", "C", "L", "chair"]}}},
            ],
        },
        {
            "people_count": 3,
            "cases": [
                {"luggage_amount": "S", "instruction": {"seats": {
                    "1": ["x", "M", "F", "chair"], "2": ["x", "M", "F", "chair"],
                    "3": ["x", "M", "F", "chair"], "4": ["x", "C", "L", "storage"]}}},
                {"luggage_amount": "M", "instruction": {"seats": {
                    "1": ["x", "M", "F", "chair"], "2": ["x", "M", "F", "chair"],
                    "3": ["y", "A", "F", "chair"], "4": ["x", "C", "L", "storage"]}}},
                {"luggage_amount": "L", "instruction": {"seats": {
                    "1": ["x", "A", "R", "chair"], "2": ["x", "C", "L", "chair"],
                    "3": ["x", "A", "R", "chair"], "4": ["x", "C", "L", "storage"]}}},
            ],
        },
        {
            "people_count": 4,
            "cases": [
                {"luggage_amount": "S", "instruction": {"seats": {
                    "1": ["x", "M", "F", "chair"], "2": ["x", "M", "F", "chair"],
                    "3": ["x", "M", "F", "chair"], "4": ["x", "M", "F", "chair"]}}},
                {"luggage_amount": "M", "instruction": {"seats": {
                    "1": ["x", "M", "F", "chair"], "2": ["x", "M", "F", "chair"],
                    "3": ["y", "A", "F", "chair"], "4": ["y", "A", "F", "chair"]}}},
                {"luggage_amount": "L", "instruction": {"seats": {
                    "1": ["x", "A", "F", "chair"], "2": ["x", "C", "F", "chair"],
                    "3": ["x", "A", "F", "chair"], "4": ["x", "C", "F", "chair"]}}},
            ],
        },
    ]
}

def _build_option_lookup(opt: dict) -> Dict[int, Dict[str, Dict[str, List[str]]]]:
    tbl: Dict[int, Dict[str, Dict[str, List[str]]]] = {}
    for block in opt["option_list"]:
        pc = block["people_count"]
        tbl[pc] = {}
        for case in block["cases"]:
            size = case["luggage_amount"]
            tbl[pc][size] = case["instruction"]["seats"]
    return tbl

OPTION_LOOKUP = _build_option_lookup(OPTION_LIST)

# ===================== chain3 task_sequence (15개) =====================
TASK_SEQ_LOOKUP: Dict[int, Dict[str, Dict[str, List[str]]]] = {
    0: {
        "S": {"1": ["fold(1)", "seat_rotate(90)", "move_on_rail(A)"],
              "2": ["fold(2)", "seat_rotate(270)", "move_on_rail(C)"],
              "3": ["fold(3)", "seat_rotate(90)", "move_on_rail(A)"],
              "4": ["fold(4)", "seat_rotate(270)", "move_on_rail(C)"]},
        "M": {"1": ["fold(1)", "seat_rotate(90)", "move_on_rail(A)"],
              "2": ["fold(2)", "seat_rotate(270)", "move_on_rail(C)"],
              "3": ["fold(3)", "seat_rotate(90)", "move_on_rail(A)"],
              "4": ["fold(4)", "seat_rotate(270)", "move_on_rail(C)"]},
        "L": {"1": ["fold(1)", "seat_rotate(90)", "move_on_rail(A)"],
              "2": ["fold(2)", "seat_rotate(270)", "move_on_rail(C)"],
              "3": ["fold(3)", "seat_rotate(90)", "move_on_rail(A)"],
              "4": ["fold(4)", "seat_rotate(270)", "move_on_rail(C)"]},
    },
    1: {
        "S": {"1": ["disk_rotate(90)", "fold(1)", "seat_rotate(270)", "move_on_rail(C)", "unfold(1)"],
              "2": ["fold(2)", "seat_rotate(270)", "move_on_rail(C)"],
              "3": ["fold(3)", "seat_rotate(90)", "move_on_rail(A)"],
              "4": ["fold(4)", "seat_rotate(270)", "move_on_rail(C)"]},
        "M": {"1": ["unchanged"],
              "2": ["fold(2)", "seat_rotate(270)", "move_on_rail(C)"],
              "3": ["fold(3)", "seat_rotate(90)", "move_on_rail(A)"],
              "4": ["fold(4)", "seat_rotate(270)", "move_on_rail(C)"]},
        "L": {"1": ["disk_rotate(90)", "fold(1)", "seat_rotate(90)", "move_on_rail(A)", "unfold(1)"],
              "2": ["fold(2)", "seat_rotate(270)", "move_on_rail(C)"],
              "3": ["fold(3)", "seat_rotate(90)", "move_on_rail(A)"],
              "4": ["fold(4)", "seat_rotate(270)", "move_on_rail(C)"]},
    },
    2: {
        "S": {"1": ["disk_rotate(90)", "fold(1)", "seat_rotate(270)", "move_on_rail(C)", "unfold(1)"],
              "2": ["disk_rotate(90)", "fold(2)", "seat_rotate(270)", "move_on_rail(C)", "unfold(2)"],
              "3": ["fold(3)", "seat_rotate(90)", "move_on_rail(A)"],
              "4": ["fold(4)", "seat_rotate(270)", "move_on_rail(C)"]},
        "M": {"1": ["unchanged"],
              "2": ["unchanged"],
              "3": ["fold(3)", "seat_rotate(90)", "move_on_rail(A)"],
              "4": ["fold(4)", "seat_rotate(270)", "move_on_rail(C)"]},
        "L": {"1": ["fold(1)", "seat_rotate(90)", "move_on_rail(A)"],
              "2": ["fold(2)", "seat_rotate(270)", "move_on_rail(C)", "unfold(2)"],
              "3": ["fold(3)", "seat_rotate(90)", "move_on_rail(A)"],
              "4": ["fold(4)", "seat_rotate(270)", "move_on_rail(C)", "unfold(4)"]},
    },
    3: {
        "S": {"1": ["unchanged"],
              "2": ["unchanged"],
              "3": ["unchanged"],
              "4": ["fold(4)", "seat_rotate(270)", "move_on_rail(C)"]},
        "M": {"1": ["unchanged"],
              "2": ["unchanged"],
              "3": ["disk_rotate(90)", "fold(3)", "seat_rotate(270)", "move_on_rail(A)", "unfold(3)"],
              "4": ["fold(4)", "seat_rotate(270)", "move_on_rail(C)"]},
        "L": {"1": ["fold(1)", "seat_rotate(90)", "move_on_rail(A)", "unfold(1)"],
              "2": ["fold(2)", "seat_rotate(270)", "move_on_rail(C)", "unfold(2)"],
              "3": ["fold(3)", "seat_rotate(90)", "move_on_rail(A)", "unfold(3)"],
              "4": ["fold(4)", "seat_rotate(270)", "move_on_rail(C)"]},
    },
    4: {
        "S": {"1": ["unchanged"], "2": ["unchanged"], "3": ["unchanged"], "4": ["unchanged"]},
        "M": {"1": ["unchanged"], "2": ["unchanged"],
              "3": ["disk_rotate(90)", "fold(3)", "seat_rotate(270)", "move_on_rail(A)", "unfold(3)"],
              "4": ["disk_rotate(90)", "fold(4)", "seat_rotate(270)", "move_on_rail(A)", "unfold(4)"]},
        "L": {"1": ["fold(1)", "move_on_rail(A)", "unfold(1)"],
              "2": ["fold(2)", "move_on_rail(C)", "unfold(2)"],
              "3": ["fold(3)", "move_on_rail(A)", "unfold(3)"],
              "4": ["fold(4)", "move_on_rail(C)", "unfold(4)"]},
    },
}

# ===================== chain2 정렬 동기화 (선택) =====================
def _detect_size_from_chain2(chain2_file: Path, people_count: int) -> Optional[str]:
    """chain2의 instruction을 읽어 OPTION_LOOKUP과 비교해 S/M/L 중 무엇인지 추정."""
    try:
        data = json.loads(chain2_file.read_text(encoding="utf-8"))
        seats = data["instruction"]["seats"]
    except Exception:
        return None
    for size, seats_ref in OPTION_LOOKUP.get(people_count, {}).items():
        if seats == seats_ref:
            return size
    return None

def _scenario_sizes_for_idx(scn: str, chain2_base: Path) -> List[str]:
    """기본 매핑을 사용하되, chain2에 동일 파일이 있으면 실제 인덱스-사이즈를 우선."""
    sizes = SCENARIO_SIZE_MAP_DEFAULT.get(scn, [])
    if len(sizes) <= 1:
        return sizes
    # 우선 people_0 기준으로 _1.json 확인(없으면 기본 매핑 유지)
    guess1 = _detect_size_from_chain2(chain2_base / "people_0" / f"{scn}_1.json", 0)
    guess2 = _detect_size_from_chain2(chain2_base / "people_0" / f"{scn}_2.json", 0)
    if guess1 and guess2 and guess1 != guess2:
        return [guess1, guess2]
    if guess1 and guess1 in sizes:
        # 다른 하나는 나머지
        other = [s for s in sizes if s != guess1][0]
        return [guess1, other]
    return sizes

# ===================== 포맷터: 요구 포맷을 '정확히' 재현 =====================
def dumps_task_sequence_exact(task: Dict[str, List[str]]) -> str:
    """
    정확히 아래 형태:
    {"task_sequence": { 
        "1": ["unchanged"],
        "2": ["unchanged"],
        "3": [ 
        "disk_rotate(90)",
        ...,
        "unfold(3)"
        ],
        "4": [ ... ]
    }}
    """
    lines: List[str] = []
    lines.append('{"task_sequence": { ')
    keys = ["1", "2", "3", "4"]
    for i, k in enumerate(keys):
        steps = task.get(k, ["unchanged"])
        is_last_key = (i == len(keys) - 1)
        if len(steps) == 1:
            # 예: "1": ["unchanged"],
            line = f'    "{k}": ["{steps[0]}"]'
            if not is_last_key:
                line += ","
            lines.append(line)
        else:
            # 예:
            # "3": [ 
            # "disk_rotate(90)",
            # ...
            # "unfold(3)"
            # ],
            lines.append(f'    "{k}": [ ')
            for j, s in enumerate(steps):
                comma = "," if j < len(steps) - 1 else ""
                lines.append(f'    "{s}"{comma}')
            close = "    ]" + ("," if not is_last_key else "")
            lines.append(close)
    lines.append("}}")
    return "\n".join(lines)

# ===================== 생성 로직 =====================
def ensure_people_dirs(base: Path) -> Dict[int, Path]:
    d: Dict[int, Path] = {}
    for n in PEOPLE_COUNTS:
        p = base / f"people_{n}"
        p.mkdir(parents=True, exist_ok=True)
        d[n] = p
    return d

def write_taskseq_json(dst: Path, seq: Dict[str, List[str]]) -> None:
    text = dumps_task_sequence_exact(seq)
    dst.write_text(text, encoding="utf-8")

def generate(base: Path, chain2_base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    people_dirs = ensure_people_dirs(base)

    for n in PEOPLE_COUNTS:
        for scn in SCENARIOS:
            if scn == "12":  # 보류
                print(f"[SKIP] people_{n}/{scn} (보류)")
                continue

            sizes = _scenario_sizes_for_idx(scn, chain2_base)

            if len(sizes) == 0:
                raise ValueError(f"시나리오 '{scn}'의 사이즈 매핑이 없습니다.")
            elif len(sizes) == 1:
                size = sizes[0]
                seq = TASK_SEQ_LOOKUP[n][size]
                dst = people_dirs[n] / f"{scn}.json"
                write_taskseq_json(dst, seq)
                print(f"[OK] people_{n}/{scn}.json  ← size={size}")
            else:
                for idx, size in enumerate(sizes, start=1):
                    seq = TASK_SEQ_LOOKUP[n][size]
                    dst = people_dirs[n] / f"{scn}_{idx}.json"
                    write_taskseq_json(dst, seq)
                print(f"[OK] people_{n}/{scn}_1.json, {scn}_2.json  ← sizes={sizes[0]},{sizes[1]}")

    print(f"\n완료: {base}")

# ===================== CLI =====================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="chain3 정답셋(task_sequence) 자동 생성")
    ap.add_argument("--base", type=str, default=str(DEFAULT_BASE),
                    help="chain3 생성 기준 경로 (기본: ~/Desktop/AIRL_ATM/SW/tetris/answer_sheet/chain3)")
    ap.add_argument("--chain2-base", type=str, default=str(DEFAULT_CHAIN2_BASE),
                    help="chain2 경로(존재 시 인덱스-사이즈를 실제 파일과 동기화)")
    args = ap.parse_args()

    base_path = Path(args.base).expanduser().resolve()
    chain2_path = Path(args.chain2_base).expanduser().resolve()
    generate(base_path, chain2_path)
