from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field, ValidationError

# ================== 경로 / 상수 ==================
DIR_ROOT = Path(os.path.dirname(__file__)).resolve()
DIR_SYSTEM = DIR_ROOT / "chain3_system"
DIR_PROMPT = DIR_ROOT / "chain3_prompt"
DIR_QUERY = DIR_ROOT / "chain3_query"
DIR_SCENARIO = DIR_ROOT / "chain3_scenario"
DIR_OUT = DIR_ROOT / "chain3_out"

SYSTEM_FILE = DIR_SYSTEM / "chain3_system.txt"
PROMPT_FILE_ORDER = [
    "chain3_prompt_role.txt",
    "chain3_prompt_environment.txt",
    "chain3_prompt_function.txt",
    "chain3_prompt_output_format.txt",
    "chain3_prompt_example.txt",
]
QUERY_FILE = DIR_QUERY / "chain3_query.txt"

Axis   = Literal["x", "y"]
Pos    = Literal["A", "M", "C"]
Facing = Literal["F", "R", "B", "L"]
Mode   = Literal["chair", "storage"]

DEFAULT_STATE = {"rail_axis": "x", "position": "M", "facing": "F", "mode": "chair"}

# ================== 유틸 ==================
def load_api_key() -> str:
    """tetris_secrets.json > env 순으로 로드."""
    try:
        with open(DIR_ROOT.parent / "tetris_secrets.json", "r", encoding="utf-8") as f:
            return json.load(f)["google"]["GOOGLE_API_KEY"]
    except Exception:
        return os.getenv("GOOGLE_API_KEY", "")

def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def encode_image_to_data_url(path: Path) -> str:
    import base64
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# ================== 스키마 ==================
class SeatState(BaseModel):
    rail_axis: Axis
    position: Pos
    facing: Facing
    mode: Mode

class Instruction(BaseModel):
    seats: Dict[str, List[str]]

class EnvBefore(BaseModel):
    seats: Dict[str, SeatState]

class EnvAfter(BaseModel):
    seats: Dict[str, Union[SeatState, Literal["no change"]]]

class OutputSchema(BaseModel):
    environment_before: EnvBefore
    instruction: Instruction
    task_sequence: Dict[str, List[str]] = Field(default_factory=dict)
    environment_after: EnvAfter

# ================== 모델 응답 JSON 추출 ==================
def strip_code_fences(s: str) -> str:
    t = s.strip()
    if t.startswith("```"):
        t = t[3:]
        if "\n" in t:
            t = t.split("\n", 1)[1]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()

def first_json_object(text: str, *, strict: bool = True) -> Dict[str, Any]:
    """텍스트에서 첫 번째 최상위 JSON 객체를 괄호 매칭으로 추출."""
    s = strip_code_fences(text)
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    cand = s[start:i+1]
                    try:
                        return json.loads(cand)
                    except Exception:
                        start = -1  # 다음 후보 탐색
    if strict:
        raise ValueError("No top-level JSON object found in model response.")
    return {}

# ================== 보정(프롬프트 규칙 반영) ==================
def coerce_json_object(x: Any) -> Dict[str, Any]:
    """맵 필드는 반드시 dict여야 함. 문자열이면 json.loads 시도."""
    if isinstance(x, dict):
        return {str(k): v for k, v in x.items()}
    if isinstance(x, str):
        try:
            y = json.loads(x)
            return {str(k): v for k, v in y.items()} if isinstance(y, dict) else {}
        except Exception:
            return {}
    return {}

def enforce_conformance(payload: Dict[str, Any], instruction_src: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(payload or {})
    out.setdefault("environment_before", {"seats": {}})
    out.setdefault("instruction", {"seats": {}})
    out.setdefault("environment_after", {"seats": {}})
    out.setdefault("task_sequence", {})

    out["environment_before"]["seats"] = coerce_json_object(out["environment_before"].get("seats"))
    out["instruction"]["seats"] = coerce_json_object(out["instruction"].get("seats"))
    out["environment_after"]["seats"] = coerce_json_object(out["environment_after"].get("seats"))

    # instruction.seats = 시나리오 원본과 동일
    if isinstance(instruction_src.get("seats"), dict):
        out["instruction"]["seats"] = {str(k): v for k, v in instruction_src["seats"].items()}

    eb: Dict[str, Any] = out["environment_before"]["seats"]
    ins: Dict[str, Any] = out["instruction"]["seats"]
    ea: Dict[str, Any] = out["environment_after"]["seats"]

    # ins dict → list
    for k, v in list(ins.items()):
        if isinstance(v, dict):
            ins[k] = [v.get("rail_axis"), v.get("position"), v.get("facing"), v.get("mode")]

    # 결합 키셋
    seat_ids = set(eb.keys()) | set(ins.keys()) | set(ea.keys())

    # before 누락 채움
    for sid in seat_ids:
        if sid not in eb or not isinstance(eb[sid], dict):
            eb[sid] = dict(DEFAULT_STATE)

    # after = before 키셋으로 강제
    fixed_ea: Dict[str, Any] = {}
    for sid in eb.keys():
        if sid in ins and isinstance(ins[sid], list) and len(ins[sid]) == 4:
            axis, pos, fac, mode = ins[sid]
            fixed_ea[sid] = {"rail_axis": axis, "position": pos, "facing": fac, "mode": mode}
        else:
            v = ea.get(sid, "no change")
            fixed_ea[sid] = v if v == "no change" or isinstance(v, dict) else "no change"
    out["environment_after"]["seats"] = fixed_ea

    # task_sequence 정규화
    ts = out.get("task_sequence", {})
    if not isinstance(ts, dict):
        ts = {}
    for k, v in list(ts.items()):
        if not isinstance(v, list):
            ts[k] = []
    out["task_sequence"] = ts

    return out

# ================== 프롬프트 규칙 검증 ==================
def validate_against_prompt_rules(obj: Dict[str, Any], instruction_src: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    # 1) 모든 seats 맵은 dict
    for sec in ["environment_before", "instruction", "environment_after"]:
        seats = ((obj.get(sec) or {}).get("seats"))
        if not isinstance(seats, dict):
            issues.append(f"{sec}.seats is not a JSON object")

    # 2) 키셋 동일성
    ebk = set((obj.get("environment_before") or {}).get("seats", {}).keys())
    eak = set((obj.get("environment_after") or {}).get("seats", {}).keys())
    if ebk != eak:
        issues.append("keys(environment_after.seats) != keys(environment_before.seats)")

    # 3) instruction.seats = 시나리오 원본과 동일
    if obj.get("instruction", {}).get("seats") != instruction_src.get("seats"):
        issues.append("instruction.seats does not exactly equal the given instruction (scenario).")

    return issues

def normalize_and_validate(raw_payload: Dict[str, Any],
                           instruction_src: Dict[str, Any]) -> OutputSchema:
    """보정 → Pydantic 검증 → 프롬프트 규칙 검증. 실패 시 즉시 중단."""
    payload = enforce_conformance(raw_payload, instruction_src)

    try:
        result: OutputSchema = OutputSchema.model_validate(payload)
    except ValidationError as ve:
        msgs = []
        for err in ve.errors():
            loc = ".".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "")
            msgs.append(f"{loc}: {msg}")
        raise ValueError("Pydantic validation failed:\n  - " + "\n  - ".join(msgs))

    issues = validate_against_prompt_rules(result.model_dump(), instruction_src)
    if issues:
        raise ValueError("Prompt-rule postcheck failed:\n  - " + "\n  - ".join(issues))

    return result

# ================== 출력 포맷터(예시 포맷) ==================
def _dump_oneline(o: Any) -> str:
    return json.dumps(o, ensure_ascii=False, separators=(', ', ': '))

def _sorted_items_by_key(d: Dict[str, Any]) -> List[Tuple[str, Any]]:
    def _key(k: str) -> Tuple[int, Union[int, str]]:
        return (0, int(k)) if str(k).isdigit() else (1, str(k))
    return [(str(k), d[k]) for k in sorted(d.keys(), key=_key)]

def format_chain3_json(data: Dict[str, Any]) -> str:
    lines = ['{']

    # environment_before
    eb_seats = (data.get('environment_before', {}).get('seats') or {})
    lines.append('  "environment_before": {')
    lines.append('    "seats": {')
    items = _sorted_items_by_key(eb_seats)
    for i, (sid, seat) in enumerate(items):
        comma = ',' if i < len(items) - 1 else ''
        lines.append(f'      "{sid}": {_dump_oneline(seat)}{comma}')
    lines.append('    }')
    lines.append('  },')

    # instruction
    instr_seats = (data.get('instruction', {}).get('seats') or {})
    lines.append('  "instruction": {')
    lines.append('    "seats": {')
    items = _sorted_items_by_key(instr_seats)
    for i, (sid, lst) in enumerate(items):
        comma = ',' if i < len(items) - 1 else ''
        lines.append(f'      "{sid}": {_dump_oneline(lst)}{comma}')
    lines.append('    }')
    lines.append('  },')

    # task_sequence (멀티라인 유지)
    ts = data.get('task_sequence') or {}
    ts_pretty = json.dumps(ts, ensure_ascii=False, indent=2)
    lines.append(f'  "task_sequence": {ts_pretty},')

    # environment_after
    ea_seats = (data.get('environment_after', {}).get('seats') or {})
    lines.append('  "environment_after": {')
    lines.append('    "seats": {')
    items = _sorted_items_by_key(ea_seats)
    for i, (sid, seat_or_nc) in enumerate(items):
        comma = ',' if i < len(items) - 1 else ''
        val = json.dumps(seat_or_nc, ensure_ascii=False) if seat_or_nc == "no change" else _dump_oneline(seat_or_nc)
        lines.append(f'      "{sid}": {val}{comma}')
    lines.append('    }')
    lines.append('  }')

    lines.append('}')
    return "\n".join(lines)

# ================== 프롬프트 메시지 빌드 ==================
def _SPLIT_SAFE_SPLIT(s: str) -> List[str]:
    s = s.replace("\r\n", "\n")
    s = s.replace("[user]\n", "\n[user]\n").replace("[assistant]\n", "\n[assistant]\n")
    chunks = [c for c in s.split("\n")]
    out: List[str] = []
    cur: List[str] = []
    cur_role: Optional[str] = None
    for line in chunks:
        if line.strip() == "[user]":
            if cur:
                out.append(f"__role__:{cur_role}\n" + "\n".join(cur).strip()); cur = []
            cur_role = "user"; continue
        if line.strip() == "[assistant]":
            if cur:
                out.append(f"__role__:{cur_role}\n" + "\n".join(cur).strip()); cur = []
            cur_role = "assistant"; continue
        cur.append(line)
    if cur:
        out.append(f"__role__:{cur_role}\n" + "\n".join(cur).strip())
    return out

def _PAIR_ROLE_CONTENT(chunks: List[str]) -> List[tuple[str, str]]:
    paired: List[tuple[str, str]] = []
    for c in chunks:
        if not c.strip():
            continue
        if not c.startswith("__role__:"):
            paired.append(("user", c.strip())); continue
        head, body = c.split("\n", 1) if "\n" in c else (c, "")
        role = head.replace("__role__:", "").strip() or "user"
        paired.append((role, body.strip()))
    return paired

def load_dialogue_messages(text_file: Path) -> List[Union[HumanMessage, AIMessage]]:
    raw = read_text(text_file)
    parts = [p for p in _SPLIT_SAFE_SPLIT(raw) if p.strip()]
    msgs: List[Union[HumanMessage, AIMessage]] = []
    for role, chunk in _PAIR_ROLE_CONTENT(parts):
        msgs.append(HumanMessage(content=chunk) if role == "user" else AIMessage(content=chunk))
    return msgs

def build_messages(instruction_text: str) -> List[Union[SystemMessage, HumanMessage, AIMessage]]:
    """
    - 시나리오 TXT(내용은 {"instruction": {...}})를 별도 Human 메시지로 그대로 추가
    - QUERY 프롬프트는 치환 없이 원문 그대로 추가 (모델은 직전 메시지를 instruction으로 해석)
    """
    msgs: List[Union[SystemMessage, HumanMessage, AIMessage]] = []
    msgs.append(SystemMessage(content=read_text(SYSTEM_FILE)))
    msgs += load_dialogue_messages(DIR_PROMPT / PROMPT_FILE_ORDER[0])

    img3 = DIR_PROMPT / "chain3_prompt_image_3.png"
    if img3.exists():
        msgs.append(
            HumanMessage(
                content=[
                    {"type": "text", "text": "Reference diagram."},
                    {"type": "image_url", "image_url": {"url": encode_image_to_data_url(img3)}},
                ]
            )
        )
    for fname in PROMPT_FILE_ORDER[1:]:
        msgs += load_dialogue_messages(DIR_PROMPT / fname)

    # 시나리오(= instruction 텍스트)를 그대로 한 블록으로 전달
    msgs.append(HumanMessage(content=instruction_text))

    # 쿼리 프롬프트는 치환 없이 그대로 사용
    q = read_text(QUERY_FILE)
    msgs.append(HumanMessage(content=q))
    return msgs

def find_scenario_txt(s: str) -> Path:
    rel = s.strip("/\\")
    if "/" in rel or "\\" in rel:
        cand = DIR_SCENARIO / rel
        if cand.suffix.lower() != ".txt":
            cand = cand.with_suffix(".txt")
        return cand
    if not rel.lower().endswith(".txt"):
        rel += ".txt"
    for p in ["people_1", "people_2", "people_3", "people_4"]:
        cand = DIR_SCENARIO / p / rel
        if cand.exists():
            return cand
    return DIR_SCENARIO / rel

# ================== 메인 ==================
def main():
    t_total_start = perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=True,
                        help="예: L / M / S 또는 people_2/L.txt (people_1~people_4 폴더에서 TXT 탐색)")
    args = parser.parse_args()

    scenario_path = find_scenario_txt(args.scenario).resolve()
    if not scenario_path.exists():
        raise FileNotFoundError(f"scenario file not found: {scenario_path}")

    # 시나리오 TXT(내용은 {"instruction": {...}} JSON) 읽기
    instruction_text = read_text(scenario_path).strip()
    try:
        scenario_json = json.loads(instruction_text)  # 검증/비교용
    except Exception as e:
        raise ValueError(f"Scenario text must be a JSON object like {{'instruction': ...}}: {e}")
    if not isinstance(scenario_json, dict) or "instruction" not in scenario_json:
        raise ValueError("Scenario JSON must be an object with an 'instruction' field.")
    instruction_src: Dict[str, Any] = scenario_json["instruction"]

    messages = build_messages(instruction_text)

    google_api_key = load_api_key()
    if not google_api_key:
        raise KeyError("Missing google.GOOGLE_API_KEY (file/env)")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=google_api_key,
        temperature=0.0
    )

    # 1회 호출(원시 텍스트 수신)
    t_model_start = perf_counter()
    raw_msg = llm.invoke(messages)
    t_model_end = perf_counter()
    model_latency = t_model_end - t_model_start

    raw_text = getattr(raw_msg, "content", str(raw_msg))

    # JSON 추출(없으면 즉시 실패)
    payload = first_json_object(raw_text, strict=True)

    # 보정 + 스키마 검증 + 규칙 검증 (실패 시 에러)
    result = normalize_and_validate(payload, instruction_src)

    # 직렬화(예시 포맷)
    json_text = format_chain3_json(result.model_dump())

    # 저장
    try:
        rel_parent = scenario_path.parent.relative_to(DIR_SCENARIO)
    except ValueError:
        rel_parent = Path(".")
    out_dir = DIR_OUT / rel_parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{scenario_path.stem}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(json_text)
    print(f"\nSaved: {out_file}")
    print(f"Model latency (invoke→response): {model_latency:.3f} s")
    print(f"Total runtime (start→saved): {(perf_counter() - t_total_start):.3f} s")

if __name__ == "__main__":
    main()
