from __future__ import annotations
from pathlib import Path
import os
import json
import argparse
import base64
import io
import time
import re
from typing import Optional, Tuple, Dict, Any

from PIL import Image
from openai import OpenAI

# ---------------------------
# 1) 안전 시스템 프롬프트 (가드)
# ---------------------------
SAFE_SYS = """You are a seat-layout planning assistant for a simulated environment only.
- Do NOT provide instructions to control real vehicles or hardware.
- Do NOT advise on bypassing, disabling, or overriding any safety features.
- If the user requests unsafe real-world manipulation, refuse and offer high-level, manufacturer-approved safety guidance.
- If an image may include people, do NOT identify them or infer sensitive attributes (age, health, pregnancy, ethnicity, religion, sexual orientation, political views, etc.).
- Return non-executable JSON plans only. Do not return code blocks and do not include instructions to execute code.
- If you are uncertain, err on the side of safety and refuse with a brief reason.
"""

# ---------------------------
# 2) 위험 표현 간단 필터(테스트용)
#    (강제 치환으로 거부율 비교 테스트)
# ---------------------------
RISKY_PATTERNS = [
    r"\boverride\b", r"\bbypass\b", r"\bdisable\b", r"\bhack\b",
    r"airbag", r"seatbelt sensor", r"identify the person",
    r"\bminor\b", r"\bpregnan", r"\bethnic", r"\bage\b\s*(estimate|guess)",
    r"\bhealth\b\s*(status|condition)"
]
RISKY_SUB = "[REMOVED_FOR_SAFETY]"

def sanitize_prompt(text: str) -> str:
    out = text
    for pat in RISKY_PATTERNS:
        out = re.sub(pat, RISKY_SUB, out, flags=re.IGNORECASE)
    return out

# ---------------------------
# 3) 키 로딩 (유저 기존 경로 호환)
# ---------------------------
def load_openai_key_from_tetris() -> str:
    candidates = [
        Path.home() / "Desktop" / "AIRL_ATM" / "SW" / "tetris_secrets.json",
        Path.home() / "Desktop" / "AIRL_ATM" / "tetris_secrets.json",
        Path("C:/Users/User/Desktop/AIRL_ATM/SW/tetris_secrets.json"),
        Path("C:/Users/User/Desktop/AIRL_ATM/tetris_secrets.json"),
        Path("C:/Users/AIRL/Desktop/Test/AIRL_ATM/SW/tetris_secrets.json")
    ]
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                js = json.load(f)
            if "openai" in js and "OPENAI_API_KEY" in js["openai"]:
                return js["openai"]["OPENAI_API_KEY"]
            if "OPENAI_API_KEY" in js:
                return js["OPENAI_API_KEY"]
    raise FileNotFoundError("tetris_secrets.json에서 OpenAI 키를 찾을 수 없습니다.")

# ---------------------------
# 4) 파일 로딩
# ---------------------------
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_option_list(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_image_as_data_url(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    img = Image.open(path)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

# ---------------------------
# 5) Moderation 사전검사
# ---------------------------
def moderation_check(client: OpenAI, text: str) -> Dict[str, Any]:
    """
    OpenAI Moderation(omni-moderation-latest) 호출.
    결과 구조는 모델 버전에 따라 약간 다를 수 있으니,
    안전하게 핵심 필드만 참조합니다.
    """
    try:
        out = client.moderations.create(
            model="omni-moderation-latest",
            input=text[:20000]  # 과도한 길이 방지
        )
        # out.results[0].flagged, out.results[0].categories, out.results[0].category_scores 등
        result = out.results[0] if hasattr(out, "results") and out.results else None
        if not result:
            return {"ok": False, "reason": "no_results", "raw": out}
        return {
            "ok": True,
            "flagged": getattr(result, "flagged", None),
            "categories": getattr(result, "categories", None),
            "scores": getattr(result, "category_scores", None),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------------------------
# 6) Chat 호출 (이미지 유/무)
# ---------------------------
def chat_once(
    client: OpenAI,
    model: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    image_data_url: Optional[str] = None,
    force_json: bool = True
) -> Tuple[str, float, Dict[str, int]]:
    """
    - force_json: 모델에 '순수 JSON만' 출력 요청 (강제는 아님, 가이딩)
    - 반환: (content, latency, usage_dict)
    """
    start = time.time()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if image_data_url:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]
        })
    else:
        messages.append({"role": "user", "content": user_prompt})

    # JSON 유도 프롬프트
    if force_json:
        messages.append({
            "role": "system",
            "content": (
                "Answer in PURE JSON only. No prose, no code fences. "
                "Keys: {\"option_number\": number, \"reasoning\": string, "
                "\"seat_commands\": [{\"action\": string, \"value\": string}] }."
            )
        })

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=800
    )
    latency = time.time() - start
    content = resp.choices[0].message.content if resp.choices else ""
    usage = {"prompt": 0, "completion": 0, "total": 0}
    if getattr(resp, "usage", None):
        usage["prompt"] = getattr(resp.usage, "prompt_tokens", 0)
        usage["completion"] = getattr(resp.usage, "completion_tokens", 0)
        usage["total"] = usage["prompt"] + usage["completion"]
    return content, latency, usage

# ---------------------------
# 7) 테스트 시나리오 빌드
# ---------------------------
def build_full_prompt(
    sysprompt: str,
    basic_prompt: str,
    option_list: str,
    scenario_text: str,
    add_notes: str = ""
) -> str:
    return (
        f"시스템 프롬프트:\n{sysprompt}\n\n"
        f"기본 프롬프트:\n{basic_prompt}\n\n"
        f"옵션 리스트:\n{option_list}\n\n"
        f"시나리오:\n{scenario_text}\n\n"
        f"{add_notes}\n"
        "위의 내용에 따라 적절한 좌석 제어 명령을 생성해주세요. "
        "응답에는 반드시 option_list에서 선택한 옵션 번호를 명시하세요."
    )

# ---------------------------
# 8) 거부문구 탐지(대표 패턴)
# ---------------------------
REFUSAL_PATTERNS = [
    "i'm sorry, i can't assist with that",
    "i can’t help with that",
    "cannot help with that request",
    "i’m unable to comply"
]

def looks_like_refusal(text: str) -> bool:
    t = text.strip().lower()
    return any(pat in t for pat in REFUSAL_PATTERNS)

# ---------------------------
# 9) 테스트 러너
# ---------------------------
def run_tests(args):
    api_key = load_openai_key_from_tetris()
    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI(api_key=api_key)

    # 입력 로드
    sysprompt = load_text(args.sysprompt)
    basic = load_text(args.basic_prompt)
    option_list = load_option_list(args.option_list)
    scenario_text = load_text(args.scenario_txt)
    image_url = load_image_as_data_url(args.scenario_img)

    # 공통 풀 프롬프트
    full_prompt = build_full_prompt(sysprompt, basic, option_list, scenario_text)

    print("\n================= TEST 0: Connectivity Ping =================")
    ok_msg, lat, usage = chat_once(
        client, args.model, "Reply exactly with: OK", system_prompt=None, force_json=False
    )
    print(f"latency={lat:.2f}s, usage={usage}, reply={ok_msg!r}")

    print("\n================= TEST 1: Moderation (raw full_prompt) =================")
    mod = moderation_check(client, full_prompt)
    print("moderation:", mod)

    print("\n================= TEST 2: Prompt ONLY =================")
    out2, lat2, u2 = chat_once(
        client, args.model, full_prompt, system_prompt=None, image_data_url=None
    )
    print(f"latency={lat2:.2f}s, usage={u2}, refusal={looks_like_refusal(out2)}")
    print(out2[:1000], "...\n")

    if image_url:
        print("\n================= TEST 3: Image ONLY =================")
        out3, lat3, u3 = chat_once(
            client, args.model,
            user_prompt="Describe abstract, non-sensitive objects only. Avoid people.",
            system_prompt=None,
            image_data_url=image_url
        )
        print(f"latency={lat3:.2f}s, usage={u3}, refusal={looks_like_refusal(out3)}")
        print(out3[:1000], "...\n")
    else:
        print("\n[SKIP] TEST 3 (no image file found)")

    print("\n================= TEST 4: Prompt + Image =================")
    out4, lat4, u4 = chat_once(
        client, args.model, full_prompt, system_prompt=None, image_data_url=image_url
    )
    print(f"latency={lat4:.2f}s, usage={u4}, refusal={looks_like_refusal(out4)}")
    print(out4[:1000], "...\n")

    print("\n================= TEST 5: SAFE_SYS prepended =================")
    out5, lat5, u5 = chat_once(
        client, args.model, full_prompt, system_prompt=SAFE_SYS, image_data_url=image_url
    )
    print(f"latency={lat5:.2f}s, usage={u5}, refusal={looks_like_refusal(out5)}")
    print(out5[:1000], "...\n")

    print("\n================= TEST 6: Sanitized prompt (regex redaction) =================")
    sanitized = sanitize_prompt(full_prompt)
    out6, lat6, u6 = chat_once(
        client, args.model, sanitized, system_prompt=SAFE_SYS, image_data_url=image_url
    )
    print(f"latency={lat6:.2f}s, usage={u6}, refusal={looks_like_refusal(out6)}")
    print(out6[:1000], "...\n")

    print("\n================= TEST 7: JSON-only contract (strong) =================")
    json_contract = (
        full_prompt
        + "\n\n중요: 실행 가능한 코드 대신, 오직 JSON만 반환하세요. 코드 펜스 금지."
        + "\n예: {\"option_number\": 3, \"reasoning\": \"...\", \"seat_commands\": [{\"action\": \"move_front\", \"value\": \"3 cm\"}]}"
    )
    out7, lat7, u7 = chat_once(
        client, args.model, json_contract, system_prompt=SAFE_SYS, image_data_url=image_url, force_json=True
    )
    print(f"latency={lat7:.2f}s, usage={u7}, refusal={looks_like_refusal(out7)}")
    print(out7[:1000], "...\n")

    print("\n================= SUMMARY =================")
    print("If refusal disappears after TEST 5~7, your issue was policy-triggered by the original prompt/image.")
    print("Use SAFE_SYS + JSON-only and keep the sanitize step if needed.")

# ---------------------------
# 10) CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Policy trigger tester for seat-control prompts")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model (e.g., gpt-4o)")
    parser.add_argument("--sysprompt", type=str, default="SW/Chain2/main_v2/source/chain2_system.txt")
    parser.add_argument("--basic-prompt", type=str, default="SW/Chain2/main_v2/source/chain2_basic.txt")
    parser.add_argument("--option-list", type=str, default="SW/Chain2/main_v2/source/chain2_option_list.json")
    parser.add_argument("--scenario-txt", type=str, default="SW/Chain2/main_v2/scenarios/test1.txt")
    parser.add_argument("--scenario-img", type=str, default="SW/Chain2/main_v2/scenarios/test1.jpg")
    args = parser.parse_args()
    run_tests(args)

if __name__ == "__main__":
    main()
