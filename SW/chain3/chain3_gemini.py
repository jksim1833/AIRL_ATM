from __future__ import annotations
import os
import re
import json
import argparse
import base64
from pathlib import Path
from typing import List, Dict, Any, Union

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from time import perf_counter  

# ================== 경로 상수 ( __file__ 기준 ) ==================
DIR_ROOT = Path(os.path.dirname(__file__)).resolve()
DIR_SYSTEM = DIR_ROOT / "chain3_system"
DIR_PROMPT = DIR_ROOT / "chain3_prompt"
DIR_QUERY = DIR_ROOT / "chain3_query"
DIR_SCENARIO = DIR_ROOT / "chain3_scenario"
DIR_OUT = DIR_ROOT / "chain3_out"

# 프롬프트 파일들
SYSTEM_FILE = DIR_SYSTEM / "chain3_system.txt"
# 순서: role → environment → function → output_format → example
PROMPT_FILE_ORDER = [
    "chain3_prompt_role.txt",
    "chain3_prompt_environment.txt",
    "chain3_prompt_function.txt",
    "chain3_prompt_output_format.txt",
    "chain3_prompt_example.txt",
]

QUERY_FILE = DIR_QUERY / "chain3_query.txt"

# ================== API 키 로드 ==================
with open(DIR_ROOT.parent / "tetris_secrets.json", "r", encoding="utf-8") as f:
    _cred = json.load(f)
GOOGLE_API_KEY = _cred["google"]["GOOGLE_API_KEY"]  # 이 키만 사용
if not GOOGLE_API_KEY:
    raise KeyError("Missing google.GOOGLE_API_KEY in tetris_secrets.json")

# ================== 유틸 ==================
_SPLIT = re.compile(r"\[user\]\n|\[assistant\]\n", re.MULTILINE)

def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_dialogue_messages(text_file: Path) -> List[Union[HumanMessage, AIMessage]]:
    """
    프롬프트 txt 내부의 [user]/[assistant] 마커를 파싱해
    파일 안에서의 대화 순서를 그대로 메시지로 변환한다.
    """
    raw = read_text(text_file)
    parts = [p for p in _SPILT_SAFE_SPLIT(raw) if p.strip()]  # 안전 분리(아래 함수)
    msgs: List[Union[HumanMessage, AIMessage]] = []
    # parts는 [user]와 [assistant] 블록이 교대로 온다고 가정하지 않고, 태그 기준으로 매핑
    for role, chunk in _PAIR_ROLE_CONTENT(parts):
        if role == "user":
            msgs.append(HumanMessage(content=chunk))
        else:
            msgs.append(AIMessage(content=chunk))
    return msgs

def _SPILT_SAFE_SPLIT(s: str) -> List[str]:
    """[user]/[assistant] 태그를 보존하며 분리하기 위한 헬퍼"""
    # 태그를 줄 시작 기준으로 강제 정렬
    s = s.replace("\r\n", "\n")
    # 태그 앞에 줄바꿈이 없을 수 있으니 보정
    s = s.replace("[user]\n", "\n[user]\n").replace("[assistant]\n", "\n[assistant]\n")
    chunks = [c for c in s.split("\n")]

    out: List[str] = []
    cur: List[str] = []
    cur_role: str | None = None

    for line in chunks:
        if line.strip() == "[user]":
            if cur:
                out.append(f"__role__:{cur_role}\n" + "\n".join(cur).strip())
                cur = []
            cur_role = "user"
            continue
        if line.strip() == "[assistant]":
            if cur:
                out.append(f"__role__:{cur_role}\n" + "\n".join(cur).strip())
                cur = []
            cur_role = "assistant"
            continue
        cur.append(line)

    if cur:
        out.append(f"__role__:{cur_role}\n" + "\n".join(cur).strip())

    # out 항목은 "__role__:user\n<내용>" 형태
    return out

def _PAIR_ROLE_CONTENT(chunks: List[str]) -> List[tuple[str, str]]:
    paired: List[tuple[str, str]] = []
    for c in chunks:
        if not c.strip():
            continue
        if not c.startswith("__role__:"):
            # 태그가 없으면 user로 간주
            paired.append(("user", c.strip()))
            continue
        head, body = c.split("\n", 1) if "\n" in c else (c, "")
        role = head.replace("__role__:", "").strip() or "user"
        paired.append((role, body.strip()))
    return paired

def encode_image_to_data_url(path: Path) -> str:
    """이미지를 data URL로 인코딩 (image/png 가정)"""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def build_messages(instruction: str) -> List[Union[SystemMessage, HumanMessage, AIMessage]]:
    msgs: List[Union[SystemMessage, HumanMessage, AIMessage]] = []
    # 1) System — 시스템 파일 그대로
    msgs.append(SystemMessage(content=read_text(SYSTEM_FILE)))
    # 2) role
    msgs += load_dialogue_messages(DIR_PROMPT / PROMPT_FILE_ORDER[0])

    # (이미지) — 오직 chain3_prompt_image_3.png 만 추가
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

    # 4~7) environment, function, output_format, example (각 파일 내 대화 구조 그대로)
    for fname in PROMPT_FILE_ORDER[1:]:
        msgs += load_dialogue_messages(DIR_PROMPT / fname)

    # 8) query — [INSTRUCTION] 치환, [ENVIRONMENT]는 TXT라서 제거
    q = read_text(QUERY_FILE)
    q = q.replace("[ENVIRONMENT]", "")
    q = q.replace("[INSTRUCTION]", instruction)
    msgs.append(HumanMessage(content=q))
    return msgs

def find_scenario_txt(s: str) -> Path:
    """
    --scenario 인자로 받은 문자열을 기반으로 TXT 파일을 찾는다.
    - 슬래시/백슬래시가 있으면 chain3_scenario/ 뒤의 상대경로로 간주 (확장자 보정 .txt)
    - 없으면 .txt 보정 후 people_1~people_4 폴더에서 탐색, 없으면 루트에서도 확인
    """
    rel = s.strip("/\\")
    if "/" in rel or "\\" in rel:
        cand = DIR_SCENARIO / rel
        if cand.suffix.lower() != ".txt":
            cand = cand.with_suffix(".txt")
        return cand
    # 파일명만 주어진 경우 (.txt 보정)
    if not rel.lower().endswith(".txt"):
        rel += ".txt"
    for p in ["people_1", "people_2", "people_3", "people_4"]:
        cand = DIR_SCENARIO / p / rel
        if cand.exists():
            return cand
    return DIR_SCENARIO / rel

def extract_json_from_text(text: str) -> str:  # dict 대신 str 반환
    """모델 응답에서 JSON 덩어리만 추출 (원본 형식 보존 / 표준 JSON 파싱 검사 포함)"""
    txt = text.strip()
    
    # 코드펜스 우선
    fence = "```"
    if fence in txt:
        first = txt.find(fence)
        second = txt.find(fence, first + len(fence))
        if second != -1:
            inner = txt[first + len(fence):second].strip()
            if inner.lower().startswith("json"):
                inner = inner[4:].strip()
            # 유효성 검사만 하고 원문 반환
            json.loads(inner)  # 파싱 테스트
            return inner
    
    # 전체 시도
    try:
        json.loads(txt)  # 파싱 테스트
        return txt
    except:
        pass
    
    # { ... } 추출
    start = txt.find("{")
    end = txt.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = txt[start:end+1]
        json.loads(candidate)  # 파싱 테스트
        return candidate
    
    raise ValueError("Failed to parse JSON from model output")

# ================== 메인 ==================
def main():
    t_total_start = perf_counter()  # ⬅️ 추가: 전체 실행 시작 시각

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="예: L / M / S 또는 people_2/L.txt (people_1~people_4 폴더에서 TXT 탐색)",
    )
    args = parser.parse_args()

    # 1) 시나리오 TXT 찾기
    scenario_path = find_scenario_txt(args.scenario).resolve()
    if not scenario_path.exists():
        raise FileNotFoundError(f"scenario file not found: {scenario_path}")

    # 2) TXT 파일 전체를 하나의 instruction 으로 사용
    instruction = read_text(scenario_path).strip()
    if not instruction:
        raise ValueError("Scenario TXT is empty")

    # 3) 메시지 구성
    messages = build_messages(instruction)

    # 4) LLM 호출: Gemini 2.5 Flash (추가 파라미터 미지정)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)
    t_model_start = perf_counter()  # ⬅️ 추가: 모델 호출 시작 시각
    ai_msg = llm.invoke(messages)  # AIMessage
    t_model_end = perf_counter()    # ⬅️ 추가: 모델 호출 종료 시각
    model_latency = t_model_end - t_model_start

    output_text = ai_msg.content if isinstance(ai_msg, AIMessage) else str(ai_msg)

    # 5) JSON 추출 (표준 JSON 파싱 확인)
    json_text = extract_json_from_text(output_text)

    # 6) 저장 경로: chain3_out/<people_x>/<베이스파일명>.json
    try:
        rel_parent = scenario_path.parent.relative_to(DIR_SCENARIO)
    except ValueError:
        rel_parent = Path(".")
    out_dir = DIR_OUT / rel_parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = scenario_path.stem + ".json"  # L.txt -> L.json
    out_file = out_dir / out_name

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(json_text)
    print(f"Saved: {out_file}")

    t_total_end = perf_counter()  # ⬅️ 추가: 전체 실행 종료 시각
    total_runtime = t_total_end - t_total_start

    # 추가 출력: 시간 측정 결과
    print(f"Model latency (invoke→response): {model_latency:.3f} s")
    print(f"Total runtime (start→saved): {total_runtime:.3f} s")

if __name__ == "__main__":
    main()
