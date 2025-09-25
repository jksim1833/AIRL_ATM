# chain2_set.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any

# ===================== 기본 경로 설정 =====================
# 기본값: 사용자 Desktop/AIRL_ATM/SW/tetris/answer_sheet/chain2
DEFAULT_BASE = Path.home() / "Desktop" / "AIRL_ATM" / "SW" / "tetris" / "answer_sheet" / "chain2"

# ===================== 옵션 표 (그대로 사용) =====================
OPTION_LIST = {
  "option_list": [
    {
      "people_count": 0,
      "cases": [
        {
          "luggage_amount": "S",
          "instruction": {
            "seats": {
              "1": ["x", "A", "R", "storage"],
              "2": ["x", "C", "L", "storage"],
              "3": ["x", "A", "R", "storage"],
              "4": ["x", "C", "L", "storage"]
            }
          }
        },
        {
          "luggage_amount": "M",
          "instruction": {
            "seats": {
              "1": ["x", "A", "R", "storage"],
              "2": ["x", "C", "L", "storage"],
              "3": ["x", "A", "R", "storage"],
              "4": ["x", "C", "L", "storage"]
            }
          }
        },
        {
          "luggage_amount": "L",
          "instruction": {
            "seats": {
              "1": ["x", "A", "R", "storage"],
              "2": ["x", "C", "L", "storage"],
              "3": ["x", "A", "R", "storage"],
              "4": ["x", "C", "L", "storage"]
            }
          }
        }
      ]
    },
    {
      "people_count": 1,
      "cases": [
        {
          "luggage_amount": "S",
          "instruction": {
            "seats": {
              "1": ["y", "C", "F", "chair"],
              "2": ["x", "C", "L", "storage"],
              "3": ["x", "A", "R", "storage"],
              "4": ["x", "C", "L", "storage"]
            }
          }
        },
        {
          "luggage_amount": "M",
          "instruction": {
            "seats": {
              "1": ["x", "M", "F", "chair"],
              "2": ["x", "C", "L", "storage"],
              "3": ["x", "A", "R", "storage"],
              "4": ["x", "C", "L", "storage"]
            }
          }
        },
        {
          "luggage_amount": "L",
          "instruction": {
            "seats": {
              "1": ["y", "A", "B", "chair"],
              "2": ["x", "C", "L", "storage"],
              "3": ["x", "A", "R", "storage"],
              "4": ["x", "C", "L", "storage"]
            }
          }
        }
      ]
    },
    {
      "people_count": 2,
      "cases": [
        {
          "luggage_amount": "S",
          "instruction": {
            "seats": {
              "1": ["y", "C", "F", "chair"],
              "2": ["y", "C", "F", "chair"],
              "3": ["x", "A", "R", "storage"],
              "4": ["x", "C", "L", "storage"]
            }
          }
        },
        {
          "luggage_amount": "M",
          "instruction": {
            "seats": {
              "1": ["x", "M", "F", "chair"],
              "2": ["x", "M", "F", "chair"],
              "3": ["x", "A", "R", "storage"],
              "4": ["x", "C", "L", "storage"]
            }
          }
        },
        {
          "luggage_amount": "L",
          "instruction": {
            "seats": {
              "1": ["x", "A", "R", "storage"],
              "2": ["x", "C", "L", "chair"],
              "3": ["x", "A", "R", "storage"],
              "4": ["x", "C", "L", "chair"]
            }
          }
        }
      ]
    },
    {
      "people_count": 3,
      "cases": [
        {
          "luggage_amount": "S",
          "instruction": {
            "seats": {
              "1": ["x", "M", "F", "chair"],
              "2": ["x", "M", "F", "chair"],
              "3": ["x", "M", "F", "chair"],
              "4": ["x", "C", "L", "storage"]
            }
          }
        },
        {
          "luggage_amount": "M",
          "instruction": {
            "seats": {
              "1": ["x", "M", "F", "chair"],
              "2": ["x", "M", "F", "chair"],
              "3": ["y", "A", "F", "chair"],
              "4": ["x", "C", "L", "storage"]
            }
          }
        },
        {
          "luggage_amount": "L",
          "instruction": {
            "seats": {
              "1": ["x", "A", "R", "chair"],
              "2": ["x", "C", "L", "chair"],
              "3": ["x", "A", "R", "chair"],
              "4": ["x", "C", "L", "storage"]
            }
          }
        }
      ]
    },
    {
      "people_count": 4,
      "cases": [
        {
          "luggage_amount": "S",
          "instruction": {
            "seats": {
              "1": ["x", "M", "F", "chair"],
              "2": ["x", "M", "F", "chair"],
              "3": ["x", "M", "F", "chair"],
              "4": ["x", "M", "F", "chair"]
            }
          }
        },
        {
          "luggage_amount": "M",
          "instruction": {
            "seats": {
              "1": ["x", "M", "F", "chair"],
              "2": ["x", "M", "F", "chair"],
              "3": ["y", "A", "F", "chair"],
              "4": ["y", "A", "F", "chair"]
            }
          }
        },
        {
          "luggage_amount": "L",
          "instruction": {
            "seats": {
              "1": ["x", "A", "F", "chair"],
              "2": ["x", "C", "F", "chair"],
              "3": ["x", "A", "F", "chair"],
              "4": ["x", "C", "F", "chair"]
            }
          }
        }
      ]
    }
  ]
}

# ===================== 시나리오 목록 & 사이즈 매핑 =====================
SCENARIOS: List[str] = [
  "0", "1-1", "1-2", "1-3", "1-4", "1-5", "2", "3", "4", "5",
  "6-1", "6-2", "7", "8-1", "8-2", "9", "10-1", "10-2",
  "11-1", "11-2", "12", "13", "14", "15", "16", "17-1", "17-2", "18", "19"
]

# 단일 사이즈는 리스트 1개, 복수 사이즈는 표기 순서대로 2개
SCENARIO_SIZE_MAP: Dict[str, List[str]] = {
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
  "12": [],  # 보류(파일 생성 건너뜀)
  "13": ["L"],
  "14": ["M"],
  "15": ["S"],
  "16": ["M", "L"],
  "17-1": ["M", "S"],
  "17-2": ["M", "S"],
  "18": ["L"],
  "19": ["L"],
}

# people_* 폴더 생성 순서: 0,1,2,3,4  (요구 반영)
PEOPLE_COUNTS: List[int] = [0, 1, 2, 3, 4]

# ===================== LOOKUP 테이블 =====================
def build_lookup(option_list: dict) -> Dict[int, Dict[str, dict]]:
  """
  OPTION_LIST를 조회 테이블로 변환:
  lookup[people_count][size] = instruction(dict)
  """
  table: Dict[int, Dict[str, dict]] = {}
  for block in option_list["option_list"]:
    pc = block["people_count"]
    table[pc] = {}
    for case in block["cases"]:
      size = case["luggage_amount"]
      instr = case["instruction"]
      table[pc][size] = instr
  return table

LOOKUP = build_lookup(OPTION_LIST)

# ===================== JSON 포맷터 (배열은 한 줄로) =====================
def _is_simple_list(lst: List[Any]) -> bool:
  return all(isinstance(x, (str, int, float, bool)) or x is None for x in lst)

def _dumps_inline_arrays(obj: Any, indent: int = 2, level: int = 0) -> str:
  """
  dict/object는 2칸 들여쓰기, list는 한 줄 배열로 직렬화.
  """
  sp = " " * (indent * level)
  if isinstance(obj, dict):
    items = list(obj.items())
    if not items:
      return "{}"
    out = ["{",]
    for i, (k, v) in enumerate(items):
      line = " " * (indent * (level + 1)) + json.dumps(k, ensure_ascii=False) + ": " + _dumps_inline_arrays(v, indent, level + 1)
      if i < len(items) - 1:
        line += ","
      out.append(line)
    out.append(sp + "}")
    return "\n".join(out)
  elif isinstance(obj, list):
    if _is_simple_list(obj):
      inner = ", ".join(json.dumps(x, ensure_ascii=False) for x in obj)
      return "[" + inner + "]"
    else:
      if not obj:
        return "[]"
      out = ["[",]
      for i, v in enumerate(obj):
        line = " " * (indent * (level + 1)) + _dumps_inline_arrays(v, indent, level + 1)
        if i < len(obj) - 1:
          line += ","
        out.append(line)
      out.append(sp + "]")
      return "\n".join(out)
  else:
    return json.dumps(obj, ensure_ascii=False)

# ===================== 파일 생성 로직 =====================
def write_instruction_json(dst: Path, instruction: dict) -> None:
  """
  JSON 파일을 정확한 키 구조와 줄바꿈/들여쓰기 규칙으로 저장.
  {
    "instruction": {
      "seats": {
        "1": ["x", "A", "R", "storage"],
        ...
      }
    }
  }
  """
  seats = instruction.get("seats")
  if not isinstance(seats, dict):
    raise ValueError("instruction에 'seats' 키가 없습니다.")
  payload = {
    "instruction": {
      "seats": seats
    }
  }
  text = _dumps_inline_arrays(payload, indent=2, level=0) + "\n"
  dst.write_text(text, encoding="utf-8")

def ensure_people_dirs(base: Path) -> Dict[int, Path]:
  people_dirs = {}
  for n in PEOPLE_COUNTS:
    d = base / f"people_{n}"
    d.mkdir(parents=True, exist_ok=True)
    people_dirs[n] = d
  return people_dirs

def generate(base: Path = DEFAULT_BASE) -> None:
  base.mkdir(parents=True, exist_ok=True)

  # 최상위: people_0..4 폴더 생성
  people_dirs = ensure_people_dirs(base)

  # 각 people_n 폴더에 시나리오 파일들 생성
  for n in PEOPLE_COUNTS:
    pdir = people_dirs[n]
    for scn in SCENARIOS:
      sizes = SCENARIO_SIZE_MAP.get(scn, [])
      if scn == "12":
        # 보류: 파일 생성 생략
        print(f"[SKIP JSON] people_{n}/{scn} (보류)")
        continue

      if not sizes:
        raise ValueError(f"시나리오 '{scn}'에 매핑된 사이즈가 없습니다.")

      for idx, size in enumerate(sizes, start=1):
        fname = f"{scn}.json" if len(sizes) == 1 else f"{scn}_{idx}.json"
        dst = pdir / fname

        try:
          instruction = LOOKUP[n][size]
        except KeyError:
          raise KeyError(f"옵션 매칭 실패: people_count={n}, size={size} (시나리오 {scn})")

        write_instruction_json(dst, instruction)

      if len(sizes) == 1:
        print(f"[OK] people_{n}/{scn}.json")
      else:
        print(f"[OK] people_{n}/{scn}_1.json, {scn}_2.json")

  print(f"\n완료: {base}")

# ===================== CLI =====================
if __name__ == "__main__":
  import argparse

  ap = argparse.ArgumentParser(description="chain2 정답셋 자동 생성 (배열 한 줄 출력 보장) — people_* 최상위 구조")
  ap.add_argument(
    "--base",
    type=str,
    default=str(DEFAULT_BASE),
    help="생성 기준 경로 (기본: ~/Desktop/AIRL_ATM/SW/tetris/answer_sheet/chain2)"
  )
  args = ap.parse_args()

  base_path = Path(args.base).expanduser().resolve()
  generate(base_path)
