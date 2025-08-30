# -*- coding: utf-8 -*-
"""
chain1_rt.py

오로지 '웹앱에서 실시간 촬영/업로드한 이미지'만 입력으로 쓰는 런타임 모듈.
- 콘솔 입력/시나리오 기반 이미지 탐색 없음
- 웹앱이 넘긴 image_path(실제 파일 경로)와 people(인원수)만 사용
- 원본 바이트/포맷 그대로 모델에 전달(HEIC/PNG/JPEG/WebP 등 MIME 정확 반영)
- 결과는 chain1_out/<scenario_name>.txt 에 저장 (scenario_name은 출력 파일 식별만)
"""

import os
import json
import base64
import mimetypes
from pathlib import Path
from typing import Optional
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# (선택) Pillow는 MIME 보조 감지에만 사용. 변환/재인코딩 없음.
try:
    from PIL import Image
except Exception:
    Image = None  # 없어도 확장자로 MIME 추정 가능


class LuggageAnalyzer:
    def __init__(
        self,
        scenario_name: str,
        people: Optional[int],
        image_path: Path,
    ):
        """
        - scenario_name: 출력 파일 식별용(결과 저장 경로 이름에만 사용)
        - people       : 웹앱에서 선택한 인원수(정수)
        - image_path   : 업로드된 '실제' 이미지 경로(확장자 포함, 실시간 촬영물)
        """
        if not image_path:
            raise ValueError("image_path는 필수입니다(웹앱 업로드 실제 경로).")
        self.scenario_name = scenario_name
        self.people = people
        self.image_path = Path(image_path)

        # API 키 로드 (모델 초기화 전)
        self._load_api_keys()

        # 경로
        desktop_path = Path.home() / "Desktop"
        self.base_path   = desktop_path / "AIRL_ATM" / "SW" / "chain1"
        self.prompt_path = self.base_path / "chain1_prompt" / "chain1_prompt_ver2_basic.txt"
        self.output_path = self.base_path / "chain1_out" / f"{scenario_name}.txt"

        # 프롬프트 로드
        self.system_prompt = self._load_prompt()

        # 모델 초기화
        print("🤖 모델 초기화 중...")
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        print("✅ 모델 초기화 완료")

    # -------------------- API 키/프롬프트 --------------------
    def _load_api_keys(self):
        """tetris_secrets.json 파일에서 API 키 로드"""
        possible_paths = [
            Path.home() / "Desktop" / "AIRL_ATM" / "SW" / "tetris_secrets.json",
            Path.home() / "Desktop" / "AIRL_ATM" / "tetris_secrets.json",
            Path("C:/Users/User/Desktop/AIRL_ATM/SW/tetris_secrets.json"),
            Path("C:/Users/User/Desktop/AIRL_ATM/tetris_secrets.json"),
        ]
        secrets_path = None
        for path in possible_paths:
            print(f"🔍 탐색 중: {path}")
            if path.exists():
                secrets_path = path
                print(f"✅ 파일 발견: {secrets_path}")
                break
        if not secrets_path:
            print("❌ 다음 위치에서 tetris_secrets.json 파일을 찾을 수 없습니다:")
            for path in possible_paths:
                print(f"  - {path}")
            raise FileNotFoundError("🚨 tetris_secrets.json 파일을 찾을 수 없습니다.")

        try:
            with open(secrets_path, 'r', encoding='utf-8') as f:
                secrets = json.load(f)
            os.environ["GOOGLE_API_KEY"] = secrets["google"]["GOOGLE_API_KEY"]
            print("🔑 API 키 로드 완료")
        except KeyError as e:
            raise KeyError(f"🚨 API 키를 찾을 수 없습니다: {e}")
        except json.JSONDecodeError:
            raise ValueError("🚨 tetris_secrets.json 파일 형식이 올바르지 않습니다")

    def _load_prompt(self) -> str:
        """프롬프트 파일 로드 및 중괄호 이스케이프 처리(기존 로직 유지)"""
        try:
            with open(self.prompt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # 이미 이스케이프된 {{, }} 보호
            content = content.replace('{{', '__TEMP_OPEN__').replace('}}', '__TEMP_CLOSE__')
            # 단일 {} 이스케이프
            content = content.replace('{', '{{').replace('}', '}}')
            # 복원
            content = content.replace('__TEMP_OPEN__', '{{').replace('__TEMP_CLOSE__', '}}')
            print("🔧 프롬프트 중괄호 이스케이프 처리 완료")
            return content
        except FileNotFoundError:
            raise FileNotFoundError(f"🚨 프롬프트 파일을 찾을 수 없습니다: {self.prompt_path}")

    # -------------------- MIME/인코딩 --------------------
    def _infer_mime(self, path: Path) -> str:
        """
        파일 확장자/내용으로 MIME 추정 → data URL에 정확히 반영.
        - 확장자 우선(대소문자 무시)
        - mimetypes 실패시 Pillow 포맷 확인
        - 그래도 실패시 확장자 직접 매핑
        - 최종 실패 시 application/octet-stream
        """
        mime, _ = mimetypes.guess_type(str(path))
        if mime:
            return mime  # e.g., image/png, image/heic, image/webp, image/jpeg ...

        fmt = ""
        if Image is not None:
            try:
                with Image.open(path) as im:
                    fmt = (im.format or "").lower()
            except Exception:
                fmt = ""

        # Pillow로 못 알아내면 확장자로 직접 매핑
        ext = path.suffix.lower().lstrip(".")
        table = {
            "jpeg": "image/jpeg",
            "jpg":  "image/jpeg",
            "png":  "image/png",
            "webp": "image/webp",
            "heic": "image/heic",
            "heif": "image/heif",
            "bmp":  "image/bmp",
            "gif":  "image/gif",
            "tiff": "image/tiff",
        }
        if fmt in table:
            return table[fmt]
        if ext in table:
            return table[ext]
        return "application/octet-stream"

    def _encode_image(self, image_path: Path) -> str:
        """이미지 바이트를 그대로 base64 인코딩(무변형)"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _create_chain(self):
        """LangChain 체인 생성(원본 MIME 반영)"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", [
                {
                    "type": "image_url",
                    "image_url": {
                        # 원본 MIME으로 data URL 생성 (HEIC/PNG/JPEG/WEBP 등)
                        "url": "data:{mime};base64,{image_data}"
                    }
                }
            ])
        ])
        return prompt | self.model

    # -------------------- 분석/저장 --------------------
    def analyze_image(self) -> Optional[str]:
        """이미지 분석 (원본 포맷/바이트 그대로 전달)"""
        try:
            if not self.image_path.exists():
                raise FileNotFoundError(f"🚨 이미지 파일을 찾을 수 없습니다: {self.image_path}")

            image_data = self._encode_image(self.image_path)  # 무손실
            mime = self._infer_mime(self.image_path)          # 원본 MIME

            chain = self._create_chain()

            t0 = time.perf_counter()
            result = chain.invoke({"image_data": image_data, "mime": mime})
            elapsed = time.perf_counter() - t0
            print(f"⏱️ 모델 호출~응답 시간: {elapsed:.3f}초")

            if hasattr(result, "content"):
                return result.content
            return str(result)

        except Exception as e:
            print(f"❌ 분석 오류: {e}")
            return None

    def _inject_people(self, result_text: str) -> str:
        """
        LLM JSON 출력 문자열 최상단에 {"people": <int>} 주입(기존 포맷 유지).
        실패 시 raw 보존.
        """
        import re
        text = (result_text or "").strip()

        # ```json ... ``` 코드블록 처리
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
        if m:
            text = m.group(1).strip()

        # 바깥 {}만 추출
        if not (text.startswith("{") and text.endswith("}")):
            first = text.find("{")
            last = text.rfind("}")
            if first != -1 and last != -1 and first < last:
                text = text[first:last+1]

        try:
            data = json.loads(text)
            out = {"people": int(self.people or 0)}
            if isinstance(data, dict):
                out.update(data)
            else:
                out["model_output"] = data
            return json.dumps(out, ensure_ascii=False, indent=2)
        except Exception:
            fallback = {
                "people": int(self.people or 0),
                "raw_model_output": result_text
            }
            return json.dumps(fallback, ensure_ascii=False, indent=2)

    def save_result(self, result: str):
        """결과 저장(기존 형식 유지)"""
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"💾 저장 완료: {self.output_path}")
        except Exception as e:
            print(f"❌ 저장 오류: {e}")

    def run_analysis(self) -> Optional[str]:
        """전체 분석 실행(웹앱에서 호출)"""
        print(f"🚀 === {self.scenario_name} 시나리오 분석 시작 ===")
        print(f"📸 이미지: {self.image_path}")
        print(f"📝 프롬프트: {self.prompt_path}")
        print(f"💾 출력: {self.output_path}")
        print(f"👥 인원수: {self.people}명")

        result = self.analyze_image()
        if result:
            result = self._inject_people(result)
            self.save_result(result)
            print("✅ 분석 및 저장 완료")
        else:
            print("❌ 분석 실패")

        print(f"\n🎉 === {self.scenario_name} 시나리오 분석 완료 ===")
        return result


# 본 모듈은 '웹앱에서만' 호출됩니다(콘솔 입력 제거).
if __name__ == "__main__":
    raise SystemExit("이 모듈은 웹앱에서 호출됩니다 (실시간 업로드 이미지 & people 사용).")
