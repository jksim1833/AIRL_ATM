# -*- coding: utf-8 -*-
"""
chain2.py (최적화 버전)

- System:  ~SW/Chain2/main_v7/source/sys.txt
- Human:   [텍스트] scenarios/test.txt + chain2_option_list.txt + [이미지] scenarios/test.jpg
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


def _import_safety_enums():
    """Google GenerativeAI SDK 버전 호환 안전 설정"""
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        return HarmCategory, HarmBlockThreshold
    except Exception:
        try:
            from google.generativeai.types.safety_types import HarmCategory, HarmBlockThreshold
            return HarmCategory, HarmBlockThreshold
        except Exception:
            return None, None


def _build_safety_settings():
    """안전 설정 생성"""
    HC, HBT = _import_safety_enums()
    if not (HC and HBT):
        return None

    categories = {}
    # 공통 카테고리
    for name in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_DANGEROUS_CONTENT"]:
        if hasattr(HC, name):
            categories[getattr(HC, name)] = HBT.BLOCK_NONE

    # 성적 카테고리 호환 처리
    if hasattr(HC, "HARM_CATEGORY_SEXUALLY_EXPLICIT"):
        categories[getattr(HC, "HARM_CATEGORY_SEXUALLY_EXPLICIT")] = HBT.BLOCK_NONE
    elif hasattr(HC, "HARM_CATEGORY_SEXUAL_CONTENT"):
        categories[getattr(HC, "HARM_CATEGORY_SEXUAL_CONTENT")] = HBT.BLOCK_NONE

    return categories if categories else None


class Chain2Runner:
    def __init__(self, temperature: float = 0.2):
        """
        Chain2Runner 초기화
        
        Args:
            temperature (float): 모델의 창의성 조절 (0.0~2.0)
                               - 0.0: 매우 일관성 있는 응답
                               - 0.2: 기본값 (권장)
                               - 1.0: 균형잡힌 창의성
                               - 2.0: 매우 창의적인 응답
        """
        self.temperature = temperature
        self._setup_paths()
        self._load_api_keys()
        self._load_prompts()
        self.model = self._build_model()
        self._setup_chain()

    def _setup_paths(self):
        """경로 설정"""
        current_path = Path(__file__).parent.absolute()
        
        # AIRL_ATM 폴더 탐색
        airl_atm_path = self._find_airl_atm_path(current_path)
        
        # 경로 설정
        self.root = airl_atm_path / "SW"
        self.dir_chain2_main = self.root / "Chain2" / "main_v7"
        self.dir_scenarios = self.dir_chain2_main / "scenarios"
        self.dir_source = self.dir_chain2_main / "source"
        self.dir_chain2_out = self.root / "chain2" / "chain2_out"
        
        # 파일 경로
        self.path_system = self.dir_source / "sys_v3.txt"
        self.path_option = self.dir_source / "chain2_option_list.txt"

    def _find_airl_atm_path(self, current_path: Path) -> Path:
        """AIRL_ATM 폴더 탐색"""
        # 상위 폴더에서 AIRL_ATM 찾기
        test_path = current_path
        for _ in range(5):
            if test_path.name == "AIRL_ATM":
                return test_path
            test_path = test_path.parent
            if str(test_path) == test_path.root:
                break
        
        # 하위에서 AIRL_ATM 찾기
        test_path = current_path
        while str(test_path) != test_path.root:
            airl_atm_candidate = test_path / "AIRL_ATM"
            if airl_atm_candidate.exists():
                return airl_atm_candidate
            test_path = test_path.parent
        
        # 기본 후보 경로들
        candidates = [
            Path("C:/Users/AIRL/Desktop/Test/AIRL_ATM"),
            Path("C:/Users/AIRL/Desktop/AIRL_ATM"),
            Path.home() / "Desktop" / "AIRL_ATM",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
                
        return Path.home() / "Desktop" / "AIRL_ATM"  # 기본값

    def _load_api_keys(self):
        """API 키 로드"""
        candidates = [
            self.root / "tetris_secrets.json",
            Path.home() / "Desktop" / "AIRL_ATM" / "tetris_secrets.json",
            Path(__file__).parent / "tetris_secrets.json",
        ]
        
        secrets_path = None
        for p in candidates:
            if p.exists() and p.stat().st_size > 0:
                secrets_path = p
                break
        
        if not secrets_path:
            raise FileNotFoundError("tetris_secrets.json 파일을 찾을 수 없습니다.")

        try:
            with open(secrets_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            api_key = data["google"]["GOOGLE_API_KEY"]
            if not api_key:
                raise ValueError("GOOGLE_API_KEY가 비어있습니다.")
                
            os.environ["GOOGLE_API_KEY"] = api_key
            print("[INFO] Google API 키 로드 완료")
            
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"API 키 파일 오류: {e}")

    def _load_prompts(self):
        """프롬프트 로드"""
        try:
            self.system_prompt = self._load_text_escaped(self.path_system)
        except FileNotFoundError:
            print(f"[WARNING] {self.path_system} 파일을 찾을 수 없습니다.")
            self.system_prompt = "다음 정보를 분석하여 결과를 생성하세요."
            
        try:
            self.option_text = self._read_text(self.path_option)
        except FileNotFoundError:
            print(f"[WARNING] {self.path_option} 파일을 찾을 수 없습니다.")
            self.option_text = "기본 처리 옵션을 적용합니다."

    def _setup_chain(self):
        """체인 설정"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", [
                {"type": "text", "text": "{chain1_text}"},
                {"type": "text", "text": "{option_text}"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_b64}"}},
            ]),
        ])
        self.chain = self.prompt | self.model

    @staticmethod
    def _read_text(path: Path) -> str:
        """텍스트 파일 읽기"""
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
        return path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _load_text_escaped(path: Path) -> str:
        """프롬프트 템플릿용 텍스트 로드 (중괄호 이스케이프)"""
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
        s = path.read_text(encoding="utf-8")
        # 중괄호 이스케이프 처리
        s = s.replace("{{", "__OPEN__").replace("}}", "__CLOSE__")
        s = s.replace("{", "{{").replace("}", "}}")
        s = s.replace("__OPEN__", "{{").replace("__CLOSE__", "}}")
        return s.strip()

    @staticmethod
    def _encode_image_b64(path: Path) -> str:
        """이미지를 base64로 인코딩"""
        if not path.exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_model(self) -> ChatGoogleGenerativeAI:
        """모델 생성"""
        safety_settings = _build_safety_settings()
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=self.temperature,
            safety_settings=safety_settings,
            convert_system_message_to_human=False,
        )

    def run(self, scenario_filename: str = "test", chain1_txt_filename: str = "test"):
        """메인 실행 함수"""
        # 파일명 정규화
        if not scenario_filename.lower().endswith(".jpg"):
            scenario_filename += ".jpg"
        if not chain1_txt_filename.lower().endswith(".txt"):
            chain1_txt_filename += ".txt"

        # 파일 경로 설정
        path_image = self.dir_scenarios / scenario_filename
        path_chain1_txt = self.dir_scenarios / chain1_txt_filename

        print(f"[INFO] Temperature: {self.temperature}")
        print(f"[INFO] 이미지: {path_image}")
        print(f"[INFO] 텍스트: {path_chain1_txt}")

        # 입력 로드
        image_b64 = self._encode_image_b64(path_image)
        chain1_text = self._read_text(path_chain1_txt)

        # 모델 호출
        start_time = time.perf_counter()
        result = self.chain.invoke({
            "image_b64": image_b64,
            "chain1_text": chain1_text,
            "option_text": self.option_text,
        })
        elapsed = time.perf_counter() - start_time

        # 결과 처리
        content = getattr(result, "content", str(result))
        print(f"[INFO] 응답 시간: {elapsed:.3f}초")

        # 결과 저장
        self.dir_chain2_out.mkdir(parents=True, exist_ok=True)
        out_name = Path(scenario_filename).stem + ".txt"
        out_path = self.dir_chain2_out / out_name
        out_path.write_text(content, encoding="utf-8")
        print(f"[OK] 저장 완료: {out_path}")
        return out_path


def main():
    """메인 함수"""
    print("=== Chain2 실행 ===")
    
    # Temperature는 코드에서 직접 수정 가능 (기본값: 0.2)
    TEMPERATURE = 0.2  # 필요시 이 값을 수정하세요 (0.0 ~ 2.0)
    
    # 파일명 입력
    scenario = input("시나리오 파일명(.jpg 생략 가능): ").strip()
    chain1_txt = input("텍스트 파일명(.txt 생략 가능): ").strip()

    # 실행
    runner = Chain2Runner(temperature=TEMPERATURE)
    runner.run(scenario, chain1_txt)


if __name__ == "__main__":
    main()