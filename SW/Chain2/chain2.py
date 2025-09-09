# -*- coding: utf-8 -*-
"""
chain2.py

- System:  ~/Desktop/AIRL_ATM/SW/chain2/chain2_prompt.txt (그대로)
- Human:   [텍스트] "다음 정보를 활용하여 결과를 생성하라."
           [텍스트] chain1_out/<입력>.txt (내용 그대로)
           [텍스트] chain2_option.txt (내용 그대로)
           [이미지] chain1_image/<시나리오>.jpg
- Output:  ~/Desktop/AIRL_ATM/SW/chain2/chain2_out/<시나리오>.txt
"""

import os
import json
import base64
from pathlib import Path
from typing import Optional
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ----- 안전 enum 호환 유틸 ----- #
def _import_safety_enums():
    """
    google.generativeai SDK 버전에 따른 enum import 호환 처리.
    반환: (HarmCategory, HarmBlockThreshold) 또는 (None, None)
    """
    try:
        # 최신/일반 경로
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        return HarmCategory, HarmBlockThreshold
    except Exception:
        try:
            # 일부 버전 호환 경로
            from google.generativeai.types.safety_types import HarmCategory, HarmBlockThreshold
            return HarmCategory, HarmBlockThreshold
        except Exception:
            return None, None

def _build_safety_settings():
    """
    SDK 버전에 맞춰 safety_settings(dict) 생성.
    - 공통 3개: HARASSMENT, HATE_SPEECH, DANGEROUS_CONTENT
    - 성적 카테고리: SEXUALLY_EXPLICIT 또는 SEXUAL_CONTENT 중 존재하는 것 선택
    """
    HC, HBT = _import_safety_enums()
    if not (HC and HBT):
        return None  # 안전설정 전달 생략(기본값 사용)

    categories = {}
    # 공통 카테고리
    for name in [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]:
        if hasattr(HC, name):
            categories[getattr(HC, name)] = HBT.BLOCK_NONE

    # 성적 카테고리 호환 처리
    if hasattr(HC, "HARM_CATEGORY_SEXUALLY_EXPLICIT"):
        categories[getattr(HC, "HARM_CATEGORY_SEXUALLY_EXPLICIT")] = HBT.BLOCK_NONE
    elif hasattr(HC, "HARM_CATEGORY_SEXUAL_CONTENT"):
        categories[getattr(HC, "HARM_CATEGORY_SEXUAL_CONTENT")] = HBT.BLOCK_NONE

    return categories if categories else None


class Chain2Runner:
    def __init__(self):
        # 기본 경로
        self.desktop = Path.home() / "Desktop"
        self.root = self.desktop / "AIRL_ATM" / "SW"

        # 서브 경로
        self.dir_chain1 = self.root / "chain1"
        self.dir_chain2 = self.root / "chain2"
        self.dir_chain1_image = self.dir_chain1 / "chain1_image"
        self.dir_chain1_out = self.dir_chain1 / "chain1_out"
        self.dir_chain2_out = self.dir_chain2 / "chain2_out"

        # 파일 경로
        self.path_system = self.dir_chain2 / "chain2_prompt.txt"
        self.path_option = self.dir_chain2 / "chain2_option.txt"

        # API 키 로드
        self._load_api_keys()

        # 프롬프트 로드
        self.system_prompt = self._load_text_escaped(self.path_system)
        self.option_text = self._read_text(self.path_option)

        # 모델 준비 (안전설정: 버전 호환)
        self.model = self._build_model()

        # 프롬프트 템플릿 구성
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", [
                {"type": "text", "text": "다음 정보를 활용하여 결과를 생성하라."},
                {"type": "text", "text": "{chain1_text}"},
                {"type": "text", "text": "{option_text}"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_b64}"}},
            ]),
        ])

        self.chain = self.prompt | self.model

    # ------------------ 유틸 ------------------ #
    def _load_api_keys(self):
        candidates = [
            self.root / "tetris_secrets.json",
            self.desktop / "AIRL_ATM" / "tetris_secrets.json",
            Path("C:/Users/User/Desktop/AIRL_ATM/SW/tetris_secrets.json"),
            Path("C:/Users/User/Desktop/AIRL_ATM/tetris_secrets.json"),
        ]
        secrets_path: Optional[Path] = None
        for p in candidates:
            if p.exists():
                secrets_path = p
                break
        if not secrets_path:
            raise FileNotFoundError("tetris_secrets.json 파일을 찾을 수 없습니다.")

        with open(secrets_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        try:
            os.environ["GOOGLE_API_KEY"] = data["google"]["GOOGLE_API_KEY"]
        except KeyError:
            raise KeyError("tetris_secrets.json에 google.GOOGLE_API_KEY 키가 없습니다.")

    @staticmethod
    def _read_text(path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
        return path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _load_text_escaped(path: Path) -> str:
        """
        ChatPromptTemplate 포맷 충돌 방지를 위해 system 파일 내 단일 중괄호를 이스케이프.
        이미 이스케이프된 {{ }} 는 보존.
        """
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
        s = path.read_text(encoding="utf-8")
        s = s.replace("{{", "__OPEN__").replace("}}", "__CLOSE__")
        s = s.replace("{", "{{").replace("}", "}}")
        s = s.replace("__OPEN__", "{{").replace("__CLOSE__", "}}")
        return s.strip()

    @staticmethod
    def _encode_image_b64(path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_model(self) -> ChatGoogleGenerativeAI:
        safety_settings = _build_safety_settings()
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            safety_settings=safety_settings,         # None이면 기본값 사용
            convert_system_message_to_human=False,   # system 유지(버전에 따라 제거 가능)
            # response_mime_type="application/json", # 필요 시 켜기
        )
        return llm

    # ------------------ 실행 ------------------ #
    def run(self, scenario_filename_input: str, chain1_txt_input: str):
        # 파일명 정규화
        scenario_name = scenario_filename_input.strip()
        if not scenario_name:
            raise ValueError("시나리오 파일명을 입력하세요.")
        if not scenario_name.lower().endswith(".jpg"):
            scenario_name += ".jpg"

        chain1_txt_name = chain1_txt_input.strip()
        if not chain1_txt_name:
            raise ValueError("chain1_out 파일명을 입력하세요.")
        if not chain1_txt_name.lower().endswith(".txt"):
            chain1_txt_name += ".txt"

        # 경로
        path_image = self.dir_chain1_image / scenario_name
        path_chain1_txt = self.dir_chain1_out / chain1_txt_name

        # 입력 로드(파일 내용 그대로 사용)
        image_b64 = self._encode_image_b64(path_image)
        chain1_text = self._read_text(path_chain1_txt)

        # 호출
        t0 = time.perf_counter()
        result = self.chain.invoke({
            "image_b64": image_b64,
            "chain1_text": chain1_text,
            "option_text": self.option_text,
        })
        elapsed = time.perf_counter() - t0

        # 응답/메타데이터
        content = getattr(result, "content", str(result))
        meta = getattr(result, "response_metadata", {}) or {}
        finish_reason = meta.get("finish_reason")
        block_reason = meta.get("block_reason")
        safety_ratings = meta.get("safety_ratings")

        print(f"[INFO] 모델 응답 시간: {elapsed:.3f}초")
        if finish_reason:
            print(f"[INFO] finish_reason: {finish_reason}")
        if block_reason:
            print(f"[INFO] block_reason: {block_reason}")
        if safety_ratings:
            print(f"[INFO] safety_ratings: {safety_ratings}")

        # 저장
        self.dir_chain2_out.mkdir(parents=True, exist_ok=True)
        out_name = Path(scenario_name).stem + ".txt"
        out_path = self.dir_chain2_out / out_name
        out_path.write_text(content, encoding="utf-8")
        print(f"[OK] 저장 완료: {out_path}")
        return out_path


def main():
    print("=== chain2 실행 ===")
    scenario = input("시나리오 파일명(.jpg 생략 가능): ").strip()
    chain1_txt = input("chain1_out 파일명(.txt 생략 가능): ").strip()

    runner = Chain2Runner()
    runner.run(scenario, chain1_txt)


if __name__ == "__main__":
    main()
