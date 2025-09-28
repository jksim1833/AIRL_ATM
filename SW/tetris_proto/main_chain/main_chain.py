# main_chain.py

import os, json, re
from pathlib import Path
from typing import List, Dict, Union
from time import perf_counter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

# [경로/키]
ROOT = Path(__file__).resolve().parent                        # .../tetris/main_chain
TETRIS_ROOT = ROOT.parent                                     # .../tetris
SECRETS_JSON = TETRIS_ROOT / "tetris_secrets.json"

# [Chain1 경로]
CHAIN1_PROMPT_TXT = ROOT / "chain1_prompt" / "chain1_prompt_3.txt"

# [Chain2 경로]
CHAIN2_PROMPT_DIR = ROOT / "chain2_prompt"
CHAIN2_PROMPT_TXT = CHAIN2_PROMPT_DIR / "chain2_prompt_2.txt"
CHAIN2_OPTION_TXT = CHAIN2_PROMPT_DIR / "chain2_option.txt"

# [Chain3 경로]
CHAIN3_DIR       = ROOT / "chain3_prompt"
C3_SYSTEM_TXT    = CHAIN3_DIR / "chain3_system.txt"
C3_QUERY_TXT     = CHAIN3_DIR / "chain3_query.txt"
C3_ROLE_TXT      = CHAIN3_DIR / "chain3_prompt_role.txt"
C3_ENV_TXT       = CHAIN3_DIR / "chain3_prompt_environment_2.txt"
C3_FUNC_TXT      = CHAIN3_DIR / "chain3_prompt_function.txt"
C3_OUTFMT_TXT    = CHAIN3_DIR / "chain3_prompt_output_format.txt"
C3_EXAMPLE_TXT   = CHAIN3_DIR / "chain3_prompt_example.txt"

def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def _escape_braces(s: str) -> str:
    s = s.replace("{{","__O__").replace("}}","__C__").replace("{","{{").replace("}","}}")
    return s.replace("__O__","{{").replace("__C__","}}")

def _require_exists(p: Path, label: str):
    if not p.exists():
        raise FileNotFoundError(f"{label} 누락: {p}")

for p, label in [
    (CHAIN1_PROMPT_TXT, "chain1_prompt_3.txt"),
    (CHAIN2_PROMPT_TXT, "chain2_prompt_2.txt"),
    (CHAIN2_OPTION_TXT, "chain2_option.txt"),
    (C3_SYSTEM_TXT, "chain3_system.txt"),
    (C3_QUERY_TXT, "chain3_query.txt"),
    (C3_ROLE_TXT, "chain3_prompt_role.txt"),
    (C3_ENV_TXT, "chain3_prompt_environment_2.txt"),
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

# [LLM] — 체인별 하드코딩
chain1_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.2,
    api_key=GOOGLE_API_KEY
)

chain2_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    api_key=GOOGLE_API_KEY
)

chain3_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-image-preview",
    temperature=0.2,
    api_key=GOOGLE_API_KEY
)

# ------------------------------------------------ chain ------------------------------------------------
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
        first = text.find("{"); last = text.rfind("}")
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
            ensure_ascii=False, indent=2,
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

# chain3
_chain3_system  = _escape_braces(_read_text(C3_SYSTEM_TXT))
_chain3_role    = _escape_braces(_read_text(C3_ROLE_TXT))
_chain3_env     = _escape_braces(_read_text(C3_ENV_TXT))
_chain3_func    = _escape_braces(_read_text(C3_FUNC_TXT))
_chain3_outfmt  = _escape_braces(_read_text(C3_OUTFMT_TXT))
_chain3_example = _escape_braces(_read_text(C3_EXAMPLE_TXT))
_chain3_query   = _escape_braces(_read_text(C3_QUERY_TXT))

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

# ---------------------- chain4 변환기 ----------------------
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
    def parse_function_call(self, func_call: str) -> Dict[str, Union[str, int, None]]:
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
    def process_cell(self, function_calls: Union[str, List[str]]) -> str:
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
                encoded_pin = '0000'
            elif func_name == "disk_rotate":
                if param not in self.encoding_rules["disk_rotate"]:
                    raise ValueError(f"Invalid degree value for disk_rotate: {param}")
                encoded_pin = self.encoding_rules["disk_rotate"][param]
            elif func_name == "move_on_rail":
                if param not in self.encoding_rules["move_on_rail"]:
                    raise ValueError(f"Invalid target value for move_on_rail: {param}")
                encoded_pin = self.encoding_rules["move_on_rail"][param]
            elif func_name == "seat_rotate":
                if param not in self.encoding_rules["seat_rotate"]:
                    raise ValueError(f"Invalid degree value for seat_rotate: {param}")
                encoded_pin = self.encoding_rules["seat_rotate"][param]
            elif func_name == "unfold":
                encoded_pin = '0000'; unfold_count += 1
            elif func_name == "fold":
                encoded_pin = '0001'
            else:
                raise ValueError(f"Unknown function: {func_name}")
            total_sum += int(encoded_pin)
        final_result = total_sum - unfold_count
        return f"{final_result:04d}"
    def convert_to_16_digit(self, task_sequence: Dict[str, Union[str, List[str]]]) -> str:
        if not isinstance(task_sequence, dict):
            raise ValueError(f"task_sequence must be dict, got: {type(task_sequence)}")
        return ''.join(self.process_cell(task_sequence.get(cell, "unchanged")) for cell in ['1','2','3','4'])
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

# ---------------------- 탭 프린터 (즉시 터미널 출력) ----------------------
def _tap_print_chain1(d):
    print("\n=====================chain1_out =====================")
    print(d.get("chain1_out", ""))
    print(f"\n🕒 chain1_run_time: {d.get('chain1_run_time', 0.0):.3f}s")
    return ""

def _tap_print_chain2(d):
    print("\n=====================chain2_out =====================")
    print(d.get("chain2_out", ""))
    print(f"\n🕒 chain2_run_time: {d.get('chain2_run_time', 0.0):.3f}s")
    return ""

def _tap_print_chain3(d):
    print("\n=====================chain3_out =====================")
    print(d.get("chain3_out", ""))
    print(f"\n🕒 chain3_run_time: {d.get('chain3_run_time', 0.0):.3f}s")
    return ""

def _tap_print_chain4(d):
    print("\n=====================chain4_out =====================")
    print(d.get("chain4_out", ""))
    return ""

# =============================== LCEL 파이프라인 ===============================
_pipeline = (
    RunnablePassthrough()

    # --- chain1 ---
    .assign(_t1_start=RunnableLambda(lambda _: perf_counter()))
    .assign(chain1_out_raw=(chain1_prompt | chain1_llm | StrOutputParser()))
    .assign(chain1_out=RunnableLambda(_inject_people_value))
    .assign(chain1_run_time=RunnableLambda(lambda d: perf_counter() - d["_t1_start"]))
    .assign(_tap1=RunnableLambda(_tap_print_chain1))

    # --- chain2 ---
    .assign(chain2_image=RunnableLambda(_chain2_image_value))
    .assign(_t2_start=RunnableLambda(lambda _: perf_counter()))
    .assign(chain2_out=(chain2_prompt | chain2_llm | StrOutputParser()))
    .assign(chain2_run_time=RunnableLambda(lambda d: perf_counter() - d["_t2_start"]))
    .assign(_tap2=RunnableLambda(_tap_print_chain2))

    # --- chain3 ---
    .assign(_t3_start=RunnableLambda(lambda _: perf_counter()))
    .assign(chain3_out=(chain3_prompt | chain3_llm | StrOutputParser()))
    .assign(chain3_run_time=RunnableLambda(lambda d: perf_counter() - d["_t3_start"]))
    .assign(_tap3=RunnableLambda(_tap_print_chain3))

    # --- chain4 (변환) ---
    .assign(chain4_out=RunnableLambda(lambda d: _run_chain4_transform(d)["chain4_out"]))
    .assign(_tap4=RunnableLambda(_tap_print_chain4))
)

def _select_outputs(d: dict) -> dict:
    return {
        "chain1_out": d.get("chain1_out", ""),
        "chain2_out": d.get("chain2_out", ""),
        "chain3_out": d.get("chain3_out", ""),
        "chain4_out": d.get("chain4_out", ""),
        "chain1_run_time": d.get("chain1_run_time", 0.0),
        "chain2_run_time": d.get("chain2_run_time", 0.0),
        "chain3_run_time": d.get("chain3_run_time", 0.0),
    }

tetris_chain = _pipeline | RunnableLambda(_select_outputs)
