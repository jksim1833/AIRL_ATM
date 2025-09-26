# main_chain.py

import os, json, re
from pathlib import Path
from typing import List, Dict, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
import base64, mimetypes
from time import perf_counter  # 체인별 시간 측정용

# [경로/키]
ROOT = Path(__file__).resolve().parent
TETRIS_ROOT = ROOT.parent
SECRETS_JSON = TETRIS_ROOT / "tetris_secrets.json"

# [Chain1 경로]
CHAIN1_PROMPT_TXT = ROOT / "chain1_prompt" / "chain1_prompt.txt"

# [Chain2 경로]
CHAIN2_PROMPT_DIR = ROOT / "chain2_prompt"
CHAIN2_PROMPT_TXT = CHAIN2_PROMPT_DIR / "chain2_prompt.txt"
CHAIN2_OPTION_TXT = CHAIN2_PROMPT_DIR / "chain2_option.txt"

# [Chain3 경로]
CHAIN3_DIR = ROOT / "chain3_prompt"
C3_SYSTEM_TXT = CHAIN3_DIR / "chain3_system.txt"
C3_QUERY_TXT = CHAIN3_DIR / "chain3_query.txt"
C3_ROLE_TXT = CHAIN3_DIR / "chain3_prompt_role.txt"
C3_ENV_TXT = CHAIN3_DIR / "chain3_prompt_environment.txt"
C3_FUNC_TXT = CHAIN3_DIR / "chain3_prompt_function.txt"
C3_OUTFMT_TXT = CHAIN3_DIR / "chain3_prompt_output_format.txt"
C3_EXAMPLE_TXT = CHAIN3_DIR / "chain3_prompt_example.txt"
# NOTE: chain3_prompt_image.png 사용을 의도적으로 제거했습니다.

def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def _escape_braces(s: str) -> str:
    s = s.replace("{{","__O__").replace("}}","__C__").replace("{","{{").replace("}","}}")
    return s.replace("__O__","{{").replace("__C__","}}")

# 리소스 존재 검사 (이미지 파일 항목 제거)
def _require_exists(p: Path, label: str):
    if not p.exists():
        raise FileNotFoundError(f"{label} 누락: {p}")

for p, label in [
    (CHAIN1_PROMPT_TXT, "chain1_prompt.txt"),
    (CHAIN2_PROMPT_TXT, "chain2_prompt.txt"),
    (CHAIN2_OPTION_TXT, "chain2_option.txt"),
    (C3_SYSTEM_TXT, "chain3_system.txt"),
    (C3_QUERY_TXT, "chain3_query.txt"),
    (C3_ROLE_TXT, "chain3_prompt_role.txt"),
    (C3_ENV_TXT, "chain3_prompt_environment.txt"),
    (C3_FUNC_TXT, "chain3_prompt_function.txt"),
    (C3_OUTFMT_TXT, "chain3_prompt_output_format.txt"),
    (C3_EXAMPLE_TXT, "chain3_prompt_example.txt"),
]:
    _require_exists(p, label)

# ---- API 키 로드 ----
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY and SECRETS_JSON.exists():
    GOOGLE_API_KEY = json.loads(_read_text(SECRETS_JSON))["google"]["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY가 설정되어야 합니다(환경변수 또는 tetris_secrets.json).")

# 모델/온도
MODEL_NAME  = os.getenv("TETRIS_LLM_MODEL", "gemini-2.5-flash")
TEMPERATURE = float(os.getenv("TETRIS_LLM_TEMPERATURE", "0.2"))

# LLM 초기화
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMPERATURE, api_key=GOOGLE_API_KEY)

# chain1
_chain1_system = _escape_braces(_read_text(CHAIN1_PROMPT_TXT))
chain1_prompt = ChatPromptTemplate.from_messages([
    ("system", _chain1_system),
    MessagesPlaceholder(variable_name="user_input"),
])

def make_chain1_user_input(people_count: int, image_data_url: str) -> List[HumanMessage]:
    return [
        HumanMessage(content=f"people_count = {people_count}"),
        HumanMessage(content=[{"type":"image_url","image_url":{"url":image_data_url}}]),
    ]

def _inject_people_into_json(result_text: str, people_count: int) -> str:
    text = (result_text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if m:
        text = m.group(1).strip()
    if not (text.startswith("{") and text.endswith("}")):
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and first < last:
            text = text[first:last+1]
    try:
        data = json.loads(text)
        out = {"people": int(people_count or 0)}
        if isinstance(data, dict):
            out.update(data)
        else:
            out["model_output"] = data
        return json.dumps(out, ensure_ascii=False, indent=2)
    except Exception:
        return json.dumps(
            {"people": int(people_count or 0), "raw_model_output": result_text},
            ensure_ascii=False,
            indent=2,
        )

def _inject_people_value(inputs: dict) -> str:
    return _inject_people_into_json(
        inputs.get("chain1_out_raw", ""),
        int(inputs.get("people_count", 0)),
    )

# chain2
def _extract_chain2_image(inputs: dict) -> dict:
    msgs = inputs["user_input"]
    img_msgs = [m for m in msgs if isinstance(m.content, list)]
    if not img_msgs:
        raise ValueError("user_input에 이미지 메시지가 없습니다.")
    return {"chain2_image": [img_msgs[0]]}

def _chain2_image_value(inputs: dict):
    return _extract_chain2_image(inputs)["chain2_image"]

_chain2_system = _escape_braces(_read_text(CHAIN2_PROMPT_TXT))
_chain2_option = _escape_braces(_read_text(CHAIN2_OPTION_TXT))
chain2_prompt = ChatPromptTemplate.from_messages([
    ("system", _chain2_system),
    ("human", "{chain1_out}"),
    MessagesPlaceholder(variable_name="chain2_image"),
    ("human", _chain2_option),
])

# chain3 (이미지 placeholder 제거)
_chain3_system   = _escape_braces(_read_text(C3_SYSTEM_TXT))
_chain3_role     = _escape_braces(_read_text(C3_ROLE_TXT))
_chain3_env      = _escape_braces(_read_text(C3_ENV_TXT))
_chain3_func     = _escape_braces(_read_text(C3_FUNC_TXT))
_chain3_outfmt   = _escape_braces(_read_text(C3_OUTFMT_TXT))
_chain3_example  = _escape_braces(_read_text(C3_EXAMPLE_TXT))
_chain3_query    = _escape_braces(_read_text(C3_QUERY_TXT))

# chain3_prompt: chain3_image 관련 placeholder 제거 (이미지 사용 안 함)
chain3_prompt = ChatPromptTemplate.from_messages([
    ("system", _chain3_system),
    ("human",  _chain3_role),
    ("human",  _chain3_env),
    ("human",  _chain3_func),
    ("human",  _chain3_outfmt),
    ("human",  _chain3_example),
    ("human",  "{chain2_out}"),
    ("human",  _chain3_query),
])

VERBOSE = os.getenv("TETRIS_VERBOSE", "0") == "1"

# chain4 class (unchanged logic)
class chain4:
    def __init__(self):
        self.encoding_rules = {
            'disk_rotate': {0: '0000', 90: '0010'},
            'move_on_rail': {'M': '0000', 'A': '0100', 'C': '0200'},
            'seat_rotate': {0: '0000', 90: '1000', 180: '2000', 270: '3000'},
            'unfold': '0000',
            'fold': '0001',
            'unchanged': '0000'
        }
    def parse_function_call(self, func_call: str):
        if not func_call or not isinstance(func_call, str):
            raise ValueError(f"Invalid function call (empty): {func_call}")
        s = func_call.strip()
        if not s:
            raise ValueError(f"Invalid function call (blank): {func_call}")
        if "(" not in s and ")" not in s:
            return {"function": s, "param": None}
        m = re.match(r"^\s*(\w+)\s*\(\s*(.*?)\s*\)\s*$", s)
        if not m:
            raise ValueError(f"Invalid function call format: {func_call}")
        func_name, arg_str = m.group(1), m.group(2)
        if arg_str == "":
            param = None
        else:
            param_raw = arg_str.strip().strip('\'"')
            param = int(param_raw) if param_raw.isdigit() else param_raw
        return {"function": func_name, "param": param}
    def encode_function(self, function_data):
        func_name = function_data['function']; param = function_data['param']
        if func_name == 'disk_rotate':
            if param not in self.encoding_rules['disk_rotate']:
                raise ValueError(f"Invalid degree value for disk_rotate: {param}")
            return self.encoding_rules['disk_rotate'][param]
        elif func_name == 'move_on_rail':
            if param not in self.encoding_rules['move_on_rail']:
                raise ValueError(f"Invalid target value for move_on_rail: {param}")
            return self.encoding_rules['move_on_rail'][param]
        elif func_name == 'seat_rotate':
            if param not in self.encoding_rules['seat_rotate']:
                raise ValueError(f"Invalid degree value for seat_rotate: {param}")
            return self.encoding_rules['seat_rotate'][param]
        elif func_name == 'unfold':
            return self.encoding_rules['unfold']
        elif func_name == 'fold':
            return self.encoding_rules['fold']
        elif func_name == 'unchanged':
            return self.encoding_rules['unchanged']
        else:
            raise ValueError(f"Unknown function: {func_name}")
    def process_cell(self, function_calls):
        if function_calls is None:
            return "0000"
        if isinstance(function_calls, str):
            raw = function_calls.strip()
            if not raw:
                calls = []
            elif ";" in raw or "\n" in raw:
                calls = [x.strip() for x in re.split(r"[;\n]", raw) if x.strip()]
            else:
                calls = [raw]
        elif isinstance(function_calls, list):
            calls = [str(x).strip() for x in function_calls if str(x).strip()]
        else:
            raise ValueError(f"Cell actions must be list or str, got: {type(function_calls)}")
        total_sum = 0
        unfold_count = 0
        for func_call in calls:
            parsed = self.parse_function_call(func_call)
            func_name = parsed["function"]; param = parsed["param"]
            if func_name == "unchanged":
                encoded_pin = self.encoding_rules["unchanged"]
            elif func_name == "disk_rotate":
                if param not in self.encoding_rules["disk_rotate"]:
                    raise ValueError(f"Invalid degree value for disk_rotate: {param}")
                encoded_pin = self.encoding_rules['disk_rotate'][param]
            elif func_name == "move_on_rail":
                if param not in self.encoding_rules['move_on_rail']:
                    raise ValueError(f"Invalid target value for move_on_rail: {param}")
                encoded_pin = self.encoding_rules['move_on_rail'][param]
            elif func_name == "seat_rotate":
                if param not in self.encoding_rules['seat_rotate']:
                    raise ValueError(f"Invalid degree value for seat_rotate: {param}")
                encoded_pin = self.encoding_rules['seat_rotate'][param]
            elif func_name == "unfold":
                encoded_pin = self.encoding_rules['unfold']
                unfold_count += 1
            elif func_name == "fold":
                encoded_pin = self.encoding_rules['fold']
            else:
                raise ValueError(f"Unknown function: {func_name}")
            total_sum += int(encoded_pin)
        final_result = total_sum - unfold_count
        return f"{final_result:04d}"
    def convert_to_16_digit(self, task_sequence):
        if not isinstance(task_sequence, dict):
            raise ValueError(f"task_sequence must be dict, got: {type(task_sequence)}")
        result_parts = []
        for cell_id in ['1', '2', '3', '4']:
            seq = task_sequence.get(cell_id, "unchanged")
            cell_result = self.process_cell(seq)
            result_parts.append(cell_result)
        return ''.join(result_parts)
    def convert_from_json_string(self, json_string: str) -> str:
        try:
            data = json.loads(json_string)
            if 'task_sequence' in data:
                return self.convert_to_16_digit(data['task_sequence'])
            else:
                return self.convert_to_16_digit(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

def _extract_json_str_for_chain4(text: str) -> str:
    if not text:
        raise ValueError("chain3_out is empty.")
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S | re.I)
    if m:
        return m.group(1).strip()
    if not (t.startswith("{") and t.endswith("}")):
        first = t.find("{"); last = t.rfind("}")
        if first != -1 and last != -1 and first < last:
            t = t[first:last+1]
    return t

_chain4_converter = chain4()

def _run_chain4_transform(inputs: dict) -> dict:
    raw = inputs.get("chain3_out", "")
    json_str = _extract_json_str_for_chain4(raw)
    result16 = _chain4_converter.convert_from_json_string(json_str)
    result16 = (result16 or "").strip()
    if not result16.isdigit():
        raise ValueError(f"chain4 result is not numeric: {result16}")
    if len(result16) < 16:
        result16 = result16.rjust(16, "0")
    elif len(result16) > 16:
        result16 = result16[:16]
    return {"chain4_out": result16}

# pipeline (chain3 no image)
_pipeline = (
    RunnablePassthrough()
    .assign(chain1_out_raw=(chain1_prompt | llm | StrOutputParser()))
    .assign(chain1_out=RunnableLambda(_inject_people_value))
    .assign(chain2_image=RunnableLambda(_chain2_image_value))
    .assign(chain2_out=(chain2_prompt | llm | StrOutputParser()))
    .assign(chain3_out=(chain3_prompt | llm | StrOutputParser()))
    .assign(chain4_out=RunnableLambda(lambda inputs: _run_chain4_transform(inputs)["chain4_out"]))
)

def _select_outputs(d: dict) -> dict:
    return {
        "chain1_out": d.get("chain1_out", ""),
        "chain2_out": d.get("chain2_out", ""),
        "chain3_out": d.get("chain3_out", ""),
        "chain4_out": d.get("chain4_out", ""),
    }

tetris_chain = _pipeline | RunnableLambda(_select_outputs)

# 실행 헬퍼들
def run_chain1(user_msgs: List[HumanMessage], people_count: int, config: dict | None = None) -> dict:
    t0 = perf_counter()
    c1_raw = (chain1_prompt | llm | StrOutputParser()).invoke({"user_input": user_msgs}, config=config)
    c1 = _inject_people_into_json(c1_raw, people_count)
    t1 = perf_counter()
    return {"chain1_out": c1, "elapsed": t1 - t0}

def run_chain2(user_msgs: List[HumanMessage], chain1_out: str, config: dict | None = None) -> dict:
    t0 = perf_counter()
    chain2_image = _extract_chain2_image({"user_input": user_msgs})["chain2_image"]
    c2 = (chain2_prompt | llm | StrOutputParser()).invoke(
        {"chain1_out": chain1_out, "chain2_image": chain2_image},
        config=config,
    )
    t1 = perf_counter()
    return {"chain2_out": c2, "elapsed": t1 - t0}

def run_chain3(chain2_out: str, config: dict | None = None) -> dict:
    t0 = perf_counter()
    c3 = (chain3_prompt | llm | StrOutputParser()).invoke(
        {"chain2_out": chain2_out},
        config=config,
    )
    t1 = perf_counter()
    return {"chain3_out": c3, "elapsed": t1 - t0}

def run_chain4(chain3_out: str, config: dict | None = None) -> dict:
    t0 = perf_counter()
    json_str = _extract_json_str_for_chain4(chain3_out)
    c4 = _chain4_converter.convert_from_json_string(json_str)
    c4 = (c4 or "").strip()
    if not c4.isdigit():
        raise ValueError(f"chain4 result is not numeric: {c4}")
    if len(c4) < 16:
        c4 = c4.rjust(16, "0")
    elif len(c4) > 16:
        c4 = c4[:16]
    t1 = perf_counter()
    return {"chain4_out": c4, "elapsed": t1 - t0}
