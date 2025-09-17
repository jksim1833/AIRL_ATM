# main_chain.py

import os, json, re
from pathlib import Path
from typing import List, Dict, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain, TransformChain, SequentialChain
from langchain_core.messages import HumanMessage

# [경로/키] __file__ 기준 상대경로와 GOOGLE_API_KEY 확보
ROOT = Path(__file__).resolve().parent                        # .../AIRL_ATM/SW/tetris/main_chain
TETRIS_ROOT = ROOT.parent                                     # .../AIRL_ATM/SW/tetris
SECRETS_JSON = TETRIS_ROOT / "tetris_secrets.json"            # tetris_secrets.json

#[Chain1 경로]
CHAIN1_PROMPT_TXT = ROOT / "chain1_prompt" / "chain1_prompt.txt"

# [Chain2 경로] __file__ 기준: ./chain2_prompt/{chain2_prompt.txt, chain2_option.txt}
CHAIN2_PROMPT_DIR = ROOT / "chain2_prompt"
CHAIN2_PROMPT_TXT = CHAIN2_PROMPT_DIR / "chain2_prompt.txt"
CHAIN2_OPTION_TXT = CHAIN2_PROMPT_DIR / "chain2_option.txt"

# [Chain3 경로] __file__ 기준: ./chain3_prompt/<파일>
CHAIN3_DIR              = ROOT / "chain3_prompt"
C3_SYSTEM_TXT           = CHAIN3_DIR / "chain3_system.txt"
C3_QUERY_TXT            = CHAIN3_DIR / "chain3_query_2.txt"
C3_ROLE_TXT             = CHAIN3_DIR / "chain3_prompt_role.txt"
C3_ENV_TXT              = CHAIN3_DIR / "chain3_prompt_environment_2.txt"
C3_FUNC_TXT             = CHAIN3_DIR /  "chain3_prompt_function.txt"
C3_OUTFMT_TXT           = CHAIN3_DIR / "chain3_prompt_output_format_2.txt"
C3_EXAMPLE_TXT          = CHAIN3_DIR / "chain3_prompt_example_2.txt"
C3_IMAGE_PNG            = CHAIN3_DIR / "chain3_prompt_image.png"

def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def _escape_braces(s: str) -> str:
    s = s.replace("{{","__O__").replace("}}","__C__").replace("{","{{").replace("}","}}")
    return s.replace("__O__","{{").replace("__C__","}}")

# === 리소스 존재 fail-fast ===
def _require_exists(p: Path, label: str):
    if not p.exists():
        raise FileNotFoundError(f"{label} 누락: {p}")

for p, label in [
    (CHAIN1_PROMPT_TXT, "chain1_prompt.txt"),
    (CHAIN2_PROMPT_TXT, "chain2_prompt.txt"),
    (CHAIN2_OPTION_TXT, "chain2_option.txt"),
    (C3_SYSTEM_TXT, "chain3_system.txt"),
    (C3_QUERY_TXT, "chain3_query_2.txt"),
    (C3_ROLE_TXT, "chain3_prompt_role.txt"),
    (C3_ENV_TXT, "chain3_prompt_environment_2.txt"),
    (C3_FUNC_TXT, "chain3_prompt_function.txt"),
    (C3_OUTFMT_TXT, "chain3_prompt_output_format_2.txt"),
    (C3_EXAMPLE_TXT, "chain3_prompt_example_2.txt"),
    (C3_IMAGE_PNG, "chain3_prompt_image.png"),
]:
    _require_exists(p, label)

# ---- API 키 로드 ----
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY and SECRETS_JSON.exists():
    GOOGLE_API_KEY = json.loads(_read_text(SECRETS_JSON))["google"]["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY가 설정되어야 합니다(환경변수 또는 tetris_secrets.json).")

# === 모델/온도 환경변수로 오버라이드 가능 ===
MODEL_NAME  = os.getenv("TETRIS_LLM_MODEL", "gemini-2.5-flash")
TEMPERATURE = float(os.getenv("TETRIS_LLM_TEMPERATURE", "0.2"))

# [LLM] Gemini 2.5 Flash 초기화
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMPERATURE, api_key=GOOGLE_API_KEY)

# ------------------------------------------------ chain ------------------------------------------------
# chain 1

# [Chain1 프롬프트] chain1_prompt.txt를 SystemMessage로 그대로 사용(중괄호 이스케이프)
_chain1_system = _escape_braces(_read_text(CHAIN1_PROMPT_TXT))
chain1_prompt = ChatPromptTemplate.from_messages([
    ("system", _chain1_system),
    MessagesPlaceholder(variable_name="user_input"),  # user_input: [사람수 텍스트, 이미지 메시지]
])

# [Chain1 입력 헬퍼] 사람수/이미지를 별도의 HumanMessage 2개로 구성(이미지는 data URL 그대로 전달)
def make_chain1_user_input(people_count: int, image_data_url: str) -> List[HumanMessage]:
    return [
        HumanMessage(content=f"people_count = {people_count}"),
        HumanMessage(content=[{"type":"image_url","image_url":{"url":image_data_url}}]),
    ]

# [Chain1 정의] 입력(user_input + system) → 출력(chain1_out: JSON 문자열 기대)
chain_1 = LLMChain(
    llm=llm,
    prompt=chain1_prompt,
    output_key="chain1_out_raw",
)

# [Chain1 출력 수정] chain1_out에 people 주입 
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

def _inject_people_transform(inputs: dict) -> dict:
    return {
        "chain1_out": _inject_people_into_json(
            inputs.get("chain1_out_raw", ""),
            int(inputs.get("people_count", 0)),
        )
    }

inject_people_chain = TransformChain(
    input_variables=["chain1_out_raw", "people_count"],
    output_variables=["chain1_out"],
    transform=_inject_people_transform,
)

# chain 2

# user_input(List[HumanMessage])에서 이미지 메시지 추출 → chain2_image에 전달
def _extract_chain2_image(inputs: dict) -> dict:
    msgs = inputs["user_input"]
    img_msgs = [m for m in msgs if isinstance(m.content, list)]
    if not img_msgs:
        raise ValueError("user_input에 이미지 메시지가 없습니다.")
    return {"chain2_image": [img_msgs[0]]}

prep_chain2_from_user_input = TransformChain(
    input_variables=["user_input"],
    output_variables=["chain2_image"],
    transform=_extract_chain2_image,
)

# [Chain2 프롬프트] system=chain2_prompt.txt, human={chain1_out}+이미지+chain2_option.txt
_chain2_system = _escape_braces(_read_text(CHAIN2_PROMPT_TXT))
_chain2_option = _escape_braces(_read_text(CHAIN2_OPTION_TXT))
chain2_prompt = ChatPromptTemplate.from_messages([
    ("system", _chain2_system),
    ("human", "{chain1_out}"),
    MessagesPlaceholder(variable_name="chain2_image"),
    ("human", _chain2_option),
])

# [Chain2 정의] 입력(chain1_out + 이미지 + option + system) → 출력(chain2_out)
chain_2 = LLMChain(
    llm=llm,
    prompt=chain2_prompt,
    output_key="chain2_out",
)

# chain 3

# [Chain3 프롬프트] system=chain3_system.txt, human=role/env/func/output_format/example + {chain2_out} + query + 이미지
_chain3_system   = _escape_braces(_read_text(C3_SYSTEM_TXT))
_chain3_role     = _escape_braces(_read_text(C3_ROLE_TXT))
_chain3_env      = _escape_braces(_read_text(C3_ENV_TXT))
_chain3_func     = _escape_braces(_read_text(C3_FUNC_TXT))
_chain3_outfmt   = _escape_braces(_read_text(C3_OUTFMT_TXT))
_chain3_example  = _escape_braces(_read_text(C3_EXAMPLE_TXT))
_chain3_query    = _escape_braces(_read_text(C3_QUERY_TXT))

chain3_prompt = ChatPromptTemplate.from_messages([
    ("system", _chain3_system),      # 시스템: 체인3 전역 규칙
    ("human",  _chain3_role),        # 휴먼: 역할 설명
    ("human",  _chain3_env),         # 휴먼: 환경 정의
    ("human",  _chain3_func),        # 휴먼: 기능/규칙
    ("human",  _chain3_outfmt),      # 휴먼: 출력 포맷
    ("human",  _chain3_example),     # 휴먼: 예시
    ("human",  "{chain2_out}"),      # 휴먼: 체인2 출력(instruction JSON)
    ("human",  _chain3_query),       # 휴먼: 체인3 쿼리
    MessagesPlaceholder(variable_name="chain3_image"),  # 휴먼: 체인3 레퍼런스 이미지
])

# [Chain3 이미지 입력 헬퍼] chain3_prompt_image.png를 data URL로 읽어 HumanMessage 생성
def make_chain3_image_input() -> List[HumanMessage]:
    import base64, mimetypes
    mime, _ = mimetypes.guess_type(str(C3_IMAGE_PNG))
    if not mime: mime = "image/png"
    data_url = "data:{};base64,{}".format(mime, base64.b64encode(C3_IMAGE_PNG.read_bytes()).decode("utf-8"))
    return [HumanMessage(content=[{"type":"image_url","image_url":{"url":data_url}}])]

# Chain3는 고정 PNG 이미지를 항상 부착
def _attach_chain3_image(_: dict) -> dict:
    return {"chain3_image": make_chain3_image_input()}

prep_chain3_image = TransformChain(
    input_variables=[],
    output_variables=["chain3_image"],
    transform=_attach_chain3_image,
)

# [Chain3 정의] 입력(chain2_out + 5문서 + query + 이미지 + system) → 출력(chain3_out)
chain_3 = LLMChain(
    llm=llm,
    prompt=chain3_prompt,
    output_key="chain3_out",
)

VERBOSE = os.getenv("TETRIS_VERBOSE", "0") == "1"

# chain 4

# [Chain4 클래스 정의] 
class chain4:
    """
    Task sequence를 16자리 십진수로 변환하는 클래스
    """
    def __init__(self):
        self.encoding_rules = {
            'disk_rotate': {0: '0000', 90: '0010'},
            'move_on_rail': {'M': '0000', 'A': '0100', 'C': '0200'},
            'seat_rotate': {0: '0000', 90: '1000', 180: '2000', 270: '3000'},
            'unfold': '0000',
            'fold': '0001'
        }

    def parse_function_call(self, func_call: str) -> Dict[str, Union[str, int]]:
        pattern = r'(\w+)\((\d+)(?:,\s*(\w+))?\)'
        match = re.match(pattern, func_call.strip())
        if not match:
            raise ValueError(f"Invalid function call format: {func_call}")
        function_name = match.group(1)
        cell_id = int(match.group(2))
        param = match.group(3) if match.group(3) else None
        if param and param.isdigit():
            param = int(param)
        return {'function': function_name, 'id': cell_id, 'param': param}

    def encode_function(self, function_data: Dict[str, Union[str, int]]) -> str:
        func_name = function_data['function']
        param = function_data['param']
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
        else:
            raise ValueError(f"Unknown function: {func_name}")

    def process_cell(self, function_calls: List[str]) -> str:
        total_sum = 0
        unfold_count = 0
        for func_call in function_calls:
            parsed_func = self.parse_function_call(func_call)
            encoded_pin = self.encode_function(parsed_func)
            total_sum += int(encoded_pin)
            if parsed_func['function'] == 'unfold':
                unfold_count += 1
        final_result = total_sum - unfold_count
        return f"{final_result:04d}"

    def convert_to_16_digit(self, task_sequence: Dict[str, List[str]]) -> str:
        result_parts = []
        for cell_id in ['1', '2', '3', '4']:
            if cell_id in task_sequence:
                cell_result = self.process_cell(task_sequence[cell_id])
            else:
                cell_result = "0000"
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

# chain3_out에서 JSON만 안전 추출
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

chain_4 = TransformChain(
    input_variables=["chain3_out"],
    output_variables=["chain4_out"],
    transform=_run_chain4_transform,
)

seq_chain = SequentialChain(
    chains=[chain_1, inject_people_chain, prep_chain2_from_user_input, chain_2, prep_chain3_image, chain_3, chain_4],
    input_variables=["user_input", "people_count"],
    output_variables=["chain1_out", "chain2_out", "chain3_out", "chain4_out"],
    verbose=VERBOSE,
)
