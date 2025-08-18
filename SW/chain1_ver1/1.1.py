'''
# gpt-image-1 사용 코드

import os
import json
import base64
import mimetypes
from pathlib import Path
from typing import List, Dict, Any
import asyncio
import argparse
import aiohttp

# ---- LangChain (OpenAI only) ----
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =========================================
#   프로젝트 루트: 현재 스크립트(chain1/1.1.py) 위치
# =========================================
ROOT = Path(__file__).parent.resolve()

# 기본 ORG ID (네가 준 값). 환경변수/시크릿에 다른 값이 있으면 그걸 우선 사용.
DEFAULT_ORG_ID = "org-sFasfsh6jkHg9MzL6H7pRaDh"


# =========================
#   키 로더 (tetris_secrets)
# =========================
def load_secrets() -> Dict[str, str]:
    """
    tetris_secrets / tetris_secrets.json 파일을 찾아 OpenAI 키/ORG/PROJECT를 읽는다.
    우선순위: 현재폴더 -> ~/Desktop -> 상위폴더
    """
    names = ["tetris_secrets", "tetris_secrets.json"]
    search_dirs = [Path.cwd(), Path.home() / "Desktop", Path.cwd().parent]

    for d in search_dirs:
        for n in names:
            p = d / n
            if p.exists() and p.is_file():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    openai = data.get("openai", {}) or {}
                    return {
                        "OPENAI_API_KEY": openai.get("OPENAI_API_KEY") or "",
                        "OPENAI_ORG_ID": openai.get("OPENAI_ORG_ID")
                                         or openai.get("YOUR_ORG_ID") or "",
                        "OPENAI_PROJECT_ID": openai.get("OPENAI_PROJECT_ID") or "",
                        "_FOUND_PATH": str(p),
                    }
                except Exception as e:
                    print(f"⚠️ secrets 파일 파싱 실패({p}): {e}")
    return {"OPENAI_API_KEY": "", "OPENAI_ORG_ID": "", "OPENAI_PROJECT_ID": "", "_FOUND_PATH": ""}


# =========================
#   프롬프트 체인 (LCEL)
# =========================
def build_analysis_chain(llm: ChatOpenAI):
    """
    이미지(Data URL)를 입력받아 gpt-image-1(편집)에 전달할 '최종 지시문(영문)'만 산출
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         # === 강화된 규칙 ===
         "You are a cargo volume analysis expert. From the provided image, pick exactly ONE object "
         "whose volume is easiest to measure. You must produce a SINGLE ENGLISH sentence that will be sent "
         "to an image editing model.\n\n"
         "HARD REQUIREMENTS (must be explicitly enforced in your instruction):\n"
         "• Work on the ORIGINAL PHOTO only — do NOT redraw, regenerate, recompose, upscale, denoise, color-correct, or enhance the image.\n"
         "• Keep the image’s resolution, aspect ratio, perspective, lighting, and all pixels EXACTLY the same; keep metadata unchanged.\n"
         "• Overlay ONE red rectangular bounding box (#FF0000) with a 4-pixel stroke directly on the original photo around the chosen target.\n"
         "• No other edits or elements: no captions, stickers, borders, shadows, highlights, or changes outside the box.\n"
         "• The output must be the original photo PLUS only that single red box.\n"
         "• Your reply must be ONE concise English sentence only (no code blocks, no extra text)."),
        ("human", [
            {"type": "text", "text": "Analyze this image and output only the final instruction:"},
            {"type": "image_url", "image_url": {"url": "{data_url}", "detail": "high"}}
        ])
    ])
    return prompt | llm | StrOutputParser()


class LLMImageGenerationExperiment:
    def __init__(self, openai_api_key: str, openai_org_id: str = "", openai_project_id: str = ""):
        """
        LLM → 이미지 생성 모델 실험 초기화 (OpenAI only)
        """
        self.openai_api_key = openai_api_key
        # ORG/PROJECT 우선순위: env > 전달값 > DEFAULT
        self.openai_org_id = os.environ.get("OPENAI_ORG_ID") or openai_org_id or DEFAULT_ORG_ID
        self.openai_project_id = os.environ.get("OPENAI_PROJECT_ID") or openai_project_id or ""

        # 환경변수 반영(일부 라이브러리가 참조)
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if self.openai_org_id:
            os.environ["OPENAI_ORG_ID"] = self.openai_org_id
        if self.openai_project_id:
            os.environ["OPENAI_PROJECT_ID"] = self.openai_project_id

        # ---------- 분석 LLM들 ----------
        self.analysis_models = {
            "gpt-4.1": ChatOpenAI(
                model="gpt-4.1",
                api_key=openai_api_key,
                max_tokens=600,
                temperature=0.1,
            ),
            "gpt-4o": ChatOpenAI(
                model="gpt-4o",
                api_key=openai_api_key,
                max_tokens=600,
                temperature=0.1,
            ),
        }

        # ---------- LLM→이미지생성 모델 매핑 ----------
        self.image_generators = {
            "gpt-4.1": "gpt-image-1",
            "gpt-4o": "gpt-image-1",
        }

        # ---------- 분석 체인 ----------
        self._chains = {name: build_analysis_chain(llm)
                        for name, llm in self.analysis_models.items()}

    # ----------------- 유틸 -----------------
    @staticmethod
    def _read_image_as_data_url(image_path: str) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        # 간단히 jpeg로 취급 (필요시 확장자에 맞춰 바꿔도 됨)
        return f"data:image/jpeg;base64,{b64}"

    def _create_output_dirs(self) -> Dict[str, Dict[str, str]]:
        base_dir = ROOT / "chain1_out" / "1.1"
        out = {}
        for name in self.analysis_models.keys():
            pdir = base_dir / "output_prompt" / name
            idir = base_dir / "output_image" / name
            pdir.mkdir(parents=True, exist_ok=True)
            idir.mkdir(parents=True, exist_ok=True)
            out[name] = {"prompt_dir": str(pdir), "image_dir": str(idir)}
        return out

    # ----------------- 1) LLM 분석 -----------------
    async def _analyze(self, model_name: str, data_url: str,
                       outpaths: Dict[str, str]) -> Dict[str, Any]:
        try:
            chain = self._chains[model_name]
            prompt_text = await chain.ainvoke({"data_url": data_url})

            prompt_path = os.path.join(outpaths["prompt_dir"],
                                       f"{model_name}_generated_prompt.txt")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt_text)

            return {"ok": True, "prompt": prompt_text,
                    "prompt_file": prompt_path}
        except Exception as e:
            return {"ok": False, "error": f"{model_name} analyze error: {e}"}

    # ----------------- 2) gpt-image-1 (REST /v1/images/edits) -----------------
    async def _call_images_edits(self, prompt: str, image_path: str,
                                 field_name: str = "image") -> Dict[str, Any]:
        """
        /v1/images/edits 호출; field_name을 'image' 또는 'image[]'로 사용할 수 있게 분리
        """
        url = "https://api.openai.com/v1/images/edits"

        # 헤더 구성 (ORG/PROJECT 포함)
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "OpenAI-Organization": self.openai_org_id,
        }
        if self.openai_project_id:
            headers["OpenAI-Project"] = self.openai_project_id

        mime, _ = mimetypes.guess_type(image_path)
        if mime is None:
            mime = "image/jpeg"

        form = aiohttp.FormData()
        form.add_field("model", "gpt-image-1")
        form.add_field("prompt", prompt)
        # 🔻 요청대로: 강제 정사각 리사이즈 제거
        # form.add_field("size", "1024x1024")
        form.add_field("n", "1")

        # 이미지 파일 첨부 (정확한 MIME 필수)
        form.add_field(
            field_name,
            open(image_path, "rb"),
            filename=os.path.basename(image_path),
            content_type=mime
        )

        async with aiohttp.ClientSession() as sess:
            async with sess.post(url, headers=headers, data=form) as r:
                text = await r.text()
                ok = (r.status == 200)
                return {"ok": ok, "status": r.status, "text": text}

    async def _gen_with_gpt_image(self, prompt: str,
                                  outdir: str, tag: str,
                                  image_path: str) -> Dict[str, Any]:
        """
        먼저 'image'로 호출 → 실패 시 'image[]'로 재시도
        """
        # 1차 시도: image
        resp1 = await self._call_images_edits(prompt, image_path, field_name="image")
        if not resp1["ok"]:
            # 400 등 일부 배포에서 배열 표기 요구하는 경우 대응
            should_retry_array = (resp1["status"] in (400, 422))
            if should_retry_array:
                resp2 = await self._call_images_edits(prompt, image_path, field_name="image[]")
                if not resp2["ok"]:
                    return {"ok": False, "error": f"gpt-image-1 edits HTTP {resp2['status']}: {resp2['text']}"}
                text = resp2["text"]
            else:
                return {"ok": False, "error": f"gpt-image-1 edits HTTP {resp1['status']}: {resp1['text']}"}
        else:
            text = resp1["text"]

        # JSON 파싱
        try:
            resp_json = json.loads(text)
        except Exception:
            return {"ok": False, "error": f"invalid JSON from images/edits: {text[:300]}..."}

        try:
            b64 = resp_json["data"][0]["b64_json"]
        except Exception:
            return {"ok": False, "error": f"no b64 image in response: {resp_json}"}

        img_bytes = base64.b64decode(b64)
        Path(outdir).mkdir(parents=True, exist_ok=True)
        outpath = os.path.join(outdir, f"{tag}_gptimage1_result.png")
        with open(outpath, "wb") as f:
            f.write(img_bytes)
        return {"ok": True, "image_file": outpath}

    # ----------------- 단일 모델 처리 -----------------
    async def _process_model(self, model_name: str, data_url: str,
                             outpaths: Dict[str, str], image_path: str) -> Dict[str, Any]:
        a = await self._analyze(model_name, data_url, outpaths)
        if not a.get("ok"):
            return {"model": model_name, "status": "error",
                    "error": a.get("error")}

        gen_prompt: str = a["prompt"]
        g = await self._gen_with_gpt_image(gen_prompt, outpaths["image_dir"],
                                           model_name, image_path)

        return {
            "model": model_name,
            "image_generator": "gpt-image-1",
            "analysis_status": "success",
            "prompt_file": a["prompt_file"],
            "image_generation_status": "success" if g.get("ok") else "error",
            "image_file": g.get("image_file"),
            "error": g.get("error"),
        }

    # ----------------- 전체 실행 -----------------
    async def run(self, image_path: str | None = None,
                  image_basename: str = "1") -> List[Dict[str, Any]]:
        if image_path and os.path.isfile(image_path):
            actual = image_path
            print(f"✅ 입력 이미지: {actual}")
        else:
            search_dirs = [ROOT / "chain1_image"]
            exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
            actual = None
            for d in search_dirs:
                base = d / image_basename
                for e in exts:
                    p = base.with_suffix(e)
                    if p.exists():
                        actual = str(p)
                        print(f"✅ 입력 이미지: {actual}")
                        break
                if actual:
                    break

        if not actual:
            hint = ROOT / "chain1_image"
            try:
                listed = ", ".join(os.listdir(hint))
            except Exception:
                listed = "(목록 조회 실패)"
            return [{
                "model": "System",
                "status": "error",
                "error": f"입력 이미지를 찾을 수 없습니다. --image 로 파일 경로를 지정하거나, "
                         f"{hint}\\{image_basename}.(jpg|png|webp..) 로 두세요.\n"
                         f"📁 현재 {hint} 목록: {listed}"
            }]

        outdirs = self._create_output_dirs()
        data_url = self._read_image_as_data_url(actual)
        tasks = [self._process_model(name, data_url, outdirs[name], actual)
                 for name in self.analysis_models.keys()]
        return await asyncio.gather(*tasks, return_exceptions=True)


# ----------------- main -----------------
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="직접 지정할 이미지 경로")
    parser.add_argument("--name", type=str, help="chain1_image 안의 파일명 (확장자 제외)")
    args = parser.parse_args()

    # 1) 환경변수 → 2) secrets 파일
    openai_api_key = os.environ.get("OPENAI_API_KEY") or ""
    openai_org_id = os.environ.get("OPENAI_ORG_ID") or ""
    openai_project_id = os.environ.get("OPENAI_PROJECT_ID") or ""

    if not openai_api_key or not openai_org_id or not openai_project_id:
        secrets = load_secrets()
        if not openai_api_key:
            openai_api_key = secrets.get("OPENAI_API_KEY", "")
        # ORG: env > secrets > DEFAULT
        if not openai_org_id:
            openai_org_id = secrets.get("OPENAI_ORG_ID", "") or DEFAULT_ORG_ID
        # PROJECT: 선택사항
        if not openai_project_id:
            openai_project_id = secrets.get("OPENAI_PROJECT_ID", "")
        if secrets.get("_FOUND_PATH"):
            print(f"🔑 secrets 파일 사용: {secrets['_FOUND_PATH']}")

    if not openai_api_key:
        print("❌ OPENAI_API_KEY를 찾을 수 없습니다.")
        return

    # 환경변수에도 반영(하위 라이브러리 호환)
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["OPENAI_ORG_ID"] = openai_org_id or DEFAULT_ORG_ID
    if openai_project_id:
        os.environ["OPENAI_PROJECT_ID"] = openai_project_id

    exp = LLMImageGenerationExperiment(
        openai_api_key=openai_api_key,
        openai_org_id=openai_org_id,
        openai_project_id=openai_project_id
    )

    print("🔬 LLM→이미지 생성 실험 시작")
    print("🤖 분석 LLM: gpt-4.1, gpt-4o")
    print("🎨 이미지 생성: gpt-image-1 (OpenAI Images API, edits via REST)")
    print(f"🏷️  Organization: {exp.openai_org_id}" + (f" | Project: {exp.openai_project_id}" if exp.openai_project_id else ""))

    results = await exp.run(image_path=args.image, image_basename=args.name or "1")

    print("\n" + "=" * 80)
    print("📊 실험 결과")
    print("=" * 80)
    ok = err = 0
    for r in results:
        if isinstance(r, Exception):
            print(f"❌ 예외: {r}"); err += 1; continue
        name = r.get("model")
        print(f"\n🤖 {name} -> gpt-image-1")
        if r.get("analysis_status") == "success":
            print(f"   📝 프롬프트: {r.get('prompt_file')}")
        st = r.get("image_generation_status")
        if st == "success":
            print(f"   🖼️ 이미지: {r.get('image_file')}"); ok += 1
        else:
            print(f"   ❌ 실패: {r.get('error','unknown')}"); err += 1
    print("\n📈 요약:", f"성공 {ok} | 실패 {err}")


if __name__ == "__main__":
    asyncio.run(main())
'''
import os
import json
import base64
import mimetypes
from pathlib import Path
from typing import List, Dict, Any
import asyncio
import argparse
import aiohttp

# ---- LangChain (OpenAI only) ----
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =========================================
#   프로젝트 루트: 현재 스크립트(chain1/1.1.py) 위치
# =========================================
ROOT = Path(__file__).parent.resolve()

# 기본 ORG ID (네가 준 값). 환경변수/시크릿에 다른 값이 있으면 그걸 우선 사용.
DEFAULT_ORG_ID = "org-sFasfsh6jkHg9MzL6H7pRaDh"


# =========================
#   키 로더 (tetris_secrets)
# =========================
def load_secrets() -> Dict[str, str]:
    """
    tetris_secrets / tetris_secrets.json 파일을 찾아 OpenAI 키/ORG/PROJECT를 읽는다.
    우선순위: 상위폴더(SW) -> 현재폴더 -> ~/Desktop
    """
    names = ["tetris_secrets", "tetris_secrets.json"]
    search_dirs = [
        Path.cwd().parent,  # SW 폴더 (chain1의 상위)
        Path.cwd(),         # 현재 폴더 (chain1)
        Path.home() / "Desktop"
    ]

    print("🔍 tetris_secrets 파일 검색 중...")
    for d in search_dirs:
        print(f"   📁 검색 폴더: {d}")
        for n in names:
            p = d / n
            print(f"      🔍 확인: {p}")
            if p.exists() and p.is_file():
                print(f"      ✅ 발견: {p}")
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    openai = data.get("openai", {}) or {}
                    print(f"      📋 키 확인: OPENAI_API_KEY={'있음' if openai.get('OPENAI_API_KEY') else '없음'}")
                    print(f"      📋 키 확인: YOUR_ORG_ID={'있음' if openai.get('YOUR_ORG_ID') else '없음'}")
                    return {
                        "OPENAI_API_KEY": openai.get("OPENAI_API_KEY") or "",
                        "OPENAI_ORG_ID": openai.get("YOUR_ORG_ID") or openai.get("OPENAI_ORG_ID") or "",
                        "OPENAI_PROJECT_ID": openai.get("OPENAI_PROJECT_ID") or "",
                        "_FOUND_PATH": str(p),
                    }
                except Exception as e:
                    print(f"⚠️ secrets 파일 파싱 실패({p}): {e}")
            else:
                print(f"      ❌ 없음: {p}")
    
    print("❌ tetris_secrets.json 파일을 찾을 수 없습니다.")
    return {"OPENAI_API_KEY": "", "OPENAI_ORG_ID": "", "OPENAI_PROJECT_ID": "", "_FOUND_PATH": ""}


# =========================
#   프롬프트 체인 (LCEL)
# =========================
def build_analysis_chain(llm: ChatOpenAI):
    """
    이미지(Data URL)를 입력받아 gpt-image-1(편집)에 전달할 '최종 지시문(영문)'만 산출
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         # === 매우 강화된 규칙 ===
         "You are a cargo volume analysis expert. From the provided image, pick exactly ONE object "
         "whose volume is easiest to measure. You must produce a SINGLE ENGLISH sentence that will be sent "
         "to an image editing model.\n\n"
         "CRITICAL REQUIREMENTS (absolutely no exceptions):\n"
         "• PRESERVE EXACT ORIGINAL IMAGE: Keep every single pixel, dimension, resolution, and aspect ratio identical to the input.\n"
         "• NO RESIZING: Do not change image width, height, or dimensions in any way. Keep original size exactly.\n"
         "• NO CROPPING: Do not crop, trim, or cut any part of the original image.\n"
         "• NO ENHANCEMENT: Do not upscale, denoise, color-correct, sharpen, blur, or modify any visual properties.\n"
         "• ONLY ADD RED BOX: The ONLY change allowed is adding one red rectangular outline (#FF0000, 4px stroke) around the chosen object.\n"
         "• PRESERVE ASPECT RATIO: Maintain the exact original width-to-height ratio without any modification.\n"
         "• NO BACKGROUND CHANGES: Keep the original background, lighting, shadows, and all environmental elements unchanged.\n"
         "• PIXEL-PERFECT PRESERVATION: Every pixel outside the red box must remain exactly as in the original.\n"
         "• Your instruction must explicitly state 'preserve original dimensions and aspect ratio exactly'.\n"
         "• Your reply must be ONE concise English sentence only."),
        ("human", [
            {"type": "text", "text": "Analyze this image and output only the final instruction:"},
            {"type": "image_url", "image_url": {"url": "{data_url}", "detail": "high"}}
        ])
    ])
    return prompt | llm | StrOutputParser()


class LLMImageGenerationExperiment:
    def __init__(self, openai_api_key: str, openai_org_id: str = "", openai_project_id: str = ""):
        """
        LLM → 이미지 생성 모델 실험 초기화 (OpenAI only)
        """
        self.openai_api_key = openai_api_key
        # ORG/PROJECT 우선순위: env > 전달값 > DEFAULT
        self.openai_org_id = os.environ.get("OPENAI_ORG_ID") or openai_org_id or DEFAULT_ORG_ID
        self.openai_project_id = os.environ.get("OPENAI_PROJECT_ID") or openai_project_id or ""

        # 환경변수 반영(일부 라이브러리가 참조)
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if self.openai_org_id:
            os.environ["OPENAI_ORG_ID"] = self.openai_org_id
        if self.openai_project_id:
            os.environ["OPENAI_PROJECT_ID"] = self.openai_project_id

        # ---------- 분석 LLM들 ----------
        self.analysis_models = {
            "gpt-4.1": ChatOpenAI(
                model="gpt-4.1",
                api_key=openai_api_key,
                max_tokens=600,
                temperature=0.1,
            ),
            "gpt-4o": ChatOpenAI(
                model="gpt-4o",
                api_key=openai_api_key,
                max_tokens=600,
                temperature=0.1,
            ),
        }

        # ---------- LLM→이미지생성 모델 매핑 ----------
        self.image_generators = {
            "gpt-4.1": "gpt-image-1",
            "gpt-4o": "gpt-image-1",
        }

        # ---------- 분석 체인 ----------
        self._chains = {name: build_analysis_chain(llm)
                        for name, llm in self.analysis_models.items()}

    # ----------------- 유틸 -----------------
    @staticmethod
    def _read_image_as_data_url(image_path: str) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        # MIME 타입을 정확히 감지
        mime, _ = mimetypes.guess_type(image_path)
        if mime is None:
            mime = "image/jpeg"
        return f"data:{mime};base64,{b64}"

    def _create_output_dirs(self) -> Dict[str, Dict[str, str]]:
        base_dir = ROOT / "chain1_out" / "1.1"
        out = {}
        for name in self.analysis_models.keys():
            pdir = base_dir / "output_prompt" / name
            idir = base_dir / "output_image" / name
            pdir.mkdir(parents=True, exist_ok=True)
            idir.mkdir(parents=True, exist_ok=True)
            out[name] = {"prompt_dir": str(pdir), "image_dir": str(idir)}
        return out

    # ----------------- 1) LLM 분석 -----------------
    async def _analyze(self, model_name: str, data_url: str,
                       outpaths: Dict[str, str]) -> Dict[str, Any]:
        try:
            chain = self._chains[model_name]
            prompt_text = await chain.ainvoke({"data_url": data_url})

            prompt_path = os.path.join(outpaths["prompt_dir"],
                                       f"{model_name}_generated_prompt.txt")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt_text)

            return {"ok": True, "prompt": prompt_text,
                    "prompt_file": prompt_path}
        except Exception as e:
            return {"ok": False, "error": f"{model_name} analyze error: {e}"}

    # ----------------- 2) gpt-image-1 (REST /v1/images/edits) -----------------
    async def _call_images_edits(self, prompt: str, image_path: str,
                                 field_name: str = "image") -> Dict[str, Any]:
        """
        /v1/images/edits 호출; 원본 이미지 크기와 종횡비를 완전히 유지
        """
        url = "https://api.openai.com/v1/images/edits"

        # 헤더 구성 (ORG/PROJECT 포함)
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "OpenAI-Organization": self.openai_org_id,
        }
        if self.openai_project_id:
            headers["OpenAI-Project"] = self.openai_project_id

        mime, _ = mimetypes.guess_type(image_path)
        if mime is None:
            mime = "image/jpeg"

        # 크기 보존을 강조하는 프롬프트 추가
        enhanced_prompt = f"{prompt} Preserve original dimensions and aspect ratio exactly, do not resize or crop the image."

        form = aiohttp.FormData()
        form.add_field("model", "gpt-image-1")
        form.add_field("prompt", enhanced_prompt)
        # 🚨 중요: size 파라미터를 완전히 제거하여 원본 크기 유지
        # 정사각형 강제 리사이즈 방지
        form.add_field("n", "1")

        # 이미지 파일 첨부 (정확한 MIME 필수)
        with open(image_path, "rb") as img_file:
            form.add_field(
                field_name,
                img_file,
                filename=os.path.basename(image_path),
                content_type=mime
            )

            async with aiohttp.ClientSession() as sess:
                async with sess.post(url, headers=headers, data=form) as r:
                    text = await r.text()
                    ok = (r.status == 200)
                    return {"ok": ok, "status": r.status, "text": text}

    async def _gen_with_gpt_image(self, prompt: str,
                                  outdir: str, tag: str,
                                  image_path: str) -> Dict[str, Any]:
        """
        먼저 'image'로 호출 → 실패 시 'image[]'로 재시도
        """
        # 1차 시도: image
        resp1 = await self._call_images_edits(prompt, image_path, field_name="image")
        if not resp1["ok"]:
            # 400 등 일부 배포에서 배열 표기 요구하는 경우 대응
            should_retry_array = (resp1["status"] in (400, 422))
            if should_retry_array:
                resp2 = await self._call_images_edits(prompt, image_path, field_name="image[]")
                if not resp2["ok"]:
                    return {"ok": False, "error": f"gpt-image-1 edits HTTP {resp2['status']}: {resp2['text']}"}
                text = resp2["text"]
            else:
                return {"ok": False, "error": f"gpt-image-1 edits HTTP {resp1['status']}: {resp1['text']}"}
        else:
            text = resp1["text"]

        # JSON 파싱
        try:
            resp_json = json.loads(text)
        except Exception:
            return {"ok": False, "error": f"invalid JSON from images/edits: {text[:300]}..."}

        try:
            b64 = resp_json["data"][0]["b64_json"]
        except Exception:
            return {"ok": False, "error": f"no b64 image in response: {resp_json}"}

        img_bytes = base64.b64decode(b64)
        Path(outdir).mkdir(parents=True, exist_ok=True)
        outpath = os.path.join(outdir, f"{tag}_gptimage1_result.png")
        with open(outpath, "wb") as f:
            f.write(img_bytes)
        return {"ok": True, "image_file": outpath}

    # ----------------- 단일 모델 처리 -----------------
    async def _process_model(self, model_name: str, data_url: str,
                             outpaths: Dict[str, str], image_path: str) -> Dict[str, Any]:
        a = await self._analyze(model_name, data_url, outpaths)
        if not a.get("ok"):
            return {"model": model_name, "status": "error",
                    "error": a.get("error")}

        gen_prompt: str = a["prompt"]
        g = await self._gen_with_gpt_image(gen_prompt, outpaths["image_dir"],
                                           model_name, image_path)

        return {
            "model": model_name,
            "image_generator": "gpt-image-1",
            "analysis_status": "success",
            "prompt_file": a["prompt_file"],
            "image_generation_status": "success" if g.get("ok") else "error",
            "image_file": g.get("image_file"),
            "error": g.get("error"),
        }

    # ----------------- 전체 실행 -----------------
    async def run(self, image_path: str | None = None,
                  image_basename: str = "1") -> List[Dict[str, Any]]:
        if image_path and os.path.isfile(image_path):
            actual = image_path
            print(f"✅ 입력 이미지: {actual}")
        else:
            search_dirs = [ROOT / "chain1_image"]
            exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
            actual = None
            for d in search_dirs:
                base = d / image_basename
                for e in exts:
                    p = base.with_suffix(e)
                    if p.exists():
                        actual = str(p)
                        print(f"✅ 입력 이미지: {actual}")
                        break
                if actual:
                    break

        if not actual:
            hint = ROOT / "chain1_image"
            try:
                listed = ", ".join(os.listdir(hint))
            except Exception:
                listed = "(목록 조회 실패)"
            return [{
                "model": "System",
                "status": "error",
                "error": f"입력 이미지를 찾을 수 없습니다. --image 로 파일 경로를 지정하거나, "
                         f"{hint}\\{image_basename}.(jpg|png|webp..) 로 두세요.\n"
                         f"📁 현재 {hint} 목록: {listed}"
            }]

        outdirs = self._create_output_dirs()
        data_url = self._read_image_as_data_url(actual)
        tasks = [self._process_model(name, data_url, outdirs[name], actual)
                 for name in self.analysis_models.keys()]
        return await asyncio.gather(*tasks, return_exceptions=True)


# ----------------- main -----------------
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="직접 지정할 이미지 경로")
    parser.add_argument("--name", type=str, help="chain1_image 안의 파일명 (확장자 제외)")
    args = parser.parse_args()

    # 1) 환경변수 → 2) secrets 파일
    openai_api_key = os.environ.get("OPENAI_API_KEY") or ""
    openai_org_id = os.environ.get("OPENAI_ORG_ID") or ""
    openai_project_id = os.environ.get("OPENAI_PROJECT_ID") or ""

    if not openai_api_key or not openai_org_id or not openai_project_id:
        secrets = load_secrets()
        if not openai_api_key:
            openai_api_key = secrets.get("OPENAI_API_KEY", "")
        # ORG: env > secrets > DEFAULT
        if not openai_org_id:
            openai_org_id = secrets.get("OPENAI_ORG_ID", "") or DEFAULT_ORG_ID
        # PROJECT: 선택사항
        if not openai_project_id:
            openai_project_id = secrets.get("OPENAI_PROJECT_ID", "")
        if secrets.get("_FOUND_PATH"):
            print(f"🔑 secrets 파일 사용: {secrets['_FOUND_PATH']}")

    if not openai_api_key:
        print("❌ OPENAI_API_KEY를 찾을 수 없습니다.")
        return

    # 환경변수에도 반영(하위 라이브러리 호환)
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["OPENAI_ORG_ID"] = openai_org_id or DEFAULT_ORG_ID
    if openai_project_id:
        os.environ["OPENAI_PROJECT_ID"] = openai_project_id

    exp = LLMImageGenerationExperiment(
        openai_api_key=openai_api_key,
        openai_org_id=openai_org_id,
        openai_project_id=openai_project_id
    )

    print("🔬 LLM→이미지 생성 실험 시작")
    print("🤖 분석 LLM: gpt-4.1, gpt-4o")
    print("🎨 이미지 생성: gpt-image-1 (OpenAI Images API, edits via REST)")
    print("📐 원본 이미지 크기와 종횡비 완전 유지 (정사각형 강제 없음)")
    print(f"🏷️  Organization: {exp.openai_org_id}" + (f" | Project: {exp.openai_project_id}" if exp.openai_project_id else ""))

    results = await exp.run(image_path=args.image, image_basename=args.name or "1")

    print("\n" + "=" * 80)
    print("📊 실험 결과")
    print("=" * 80)
    ok = err = 0
    for r in results:
        if isinstance(r, Exception):
            print(f"❌ 예외: {r}"); err += 1; continue
        name = r.get("model")
        print(f"\n🤖 {name} -> gpt-image-1")
        if r.get("analysis_status") == "success":
            print(f"   📝 프롬프트: {r.get('prompt_file')}")
        st = r.get("image_generation_status")
        if st == "success":
            print(f"   🖼️ 이미지: {r.get('image_file')}"); ok += 1
        else:
            print(f"   ❌ 실패: {r.get('error','unknown')}"); err += 1
    print("\n📈 요약:", f"성공 {ok} | 실패 {err}")


if __name__ == "__main__":
    asyncio.run(main())