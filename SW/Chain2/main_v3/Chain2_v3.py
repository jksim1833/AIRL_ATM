import google.generativeai as genai
import json
import os
from PIL import Image

# 1. 여기에 YOUR_API_KEY를 본인의 실제 Gemini API 키로 변경하세요.
genai.configure(api_key="AIzaSyADmicI_5Fz92zn78h5ckHU-PwmLQPhSac")

class GeminiMultiStepAnalyzer:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model = genai.GenerativeModel(model_name)

    def _call_gemini_api(self, prompt, image=None):
        """
        Gemini API를 호출하고 응답 텍스트를 반환합니다.
        """
        try:
            content_parts = [prompt]
            if image:
                content_parts.append(image)

            safety_settings = [
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]

            response = self.model.generate_content(
                content_parts,
                safety_settings=safety_settings
            )
            
            if response.candidates and response.text:
                return response.text
            else:
                finish_reason = response.prompt_feedback.finish_reason.name if hasattr(response.prompt_feedback, 'finish_reason') else "UNKNOWN"
                raise Exception(f"API 응답이 차단되었습니다. (reason: {finish_reason})")

        except Exception as e:
            raise e

    def run_analysis(self, json_file_path, image_file_path):
        """
        단계별 분석을 실행합니다.
        """
        print("--- 단계별 분석 시작 ---")
        
        try:
            # 파일 로드
            with open(json_file_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            image = Image.open(image_file_path)

            # ===============================================
            # 1단계: 짐 크기 추론 (이미지 + JSON)
            # ===============================================
            print("\n=== [1단계] 짐 크기 추론 진행 중... ===")
            prompt_step1 = (
                "아래 JSON 데이터와 이미지를 기반으로 각 짐의 크기(S, M, L)를 추론하고, "
                "추론 결과를 **오직** JSON 형식으로만 반환해 주세요. "
                "반환 형식은 다음과 같습니다:\n"
                "```json\n"
                "{\n"
                "  \"people\": 3,\n"
                "  \"total_luggage_count\": 9,\n"
                "  \"estimated_luggage_details\": {\n"
                "    \"luggage_1\": { \"object\": \"...\", \"size\": \"L\" },\n"
                "    \"luggage_2\": { \"object\": \"...\", \"size\": \"M\" }\n"
                "  }\n"
                "}\n"
                "```\n\n"
            )
            
            estimated_json_text = self._call_gemini_api(prompt_step1 + json.dumps(original_data, ensure_ascii=False, indent=2), image)
            
            # 모델이 반환한 원본 텍스트 출력
            print("--- 모델이 반환한 원본 텍스트 ---")
            print(estimated_json_text)
            print("------------------------------")

            # JSON 파싱 시도
            try:
                estimated_data = json.loads(estimated_json_text)
            except json.JSONDecodeError:
                # JSON 형식이 아닐 경우, 응답에서 JSON 코드 블록 추출
                try:
                    start_index = estimated_json_text.find("```json") + 7
                    end_index = estimated_json_text.rfind("```")
                    json_string = estimated_json_text[start_index:end_index].strip()
                    estimated_data = json.loads(json_string)
                except Exception as e:
                    raise Exception(f"응답에서 JSON 데이터를 추출할 수 없습니다: {e}")

            print("✅ 1단계 성공: 짐 크기 추론 완료.")
            print(f"추론 결과:\n{json.dumps(estimated_data, ensure_ascii=False, indent=2)}")

            # ===============================================
            # 2단계: 총 공간 요구량 분석 (JSON만 사용)
            # ===============================================
            print("\n=== [2단계] 총 공간 요구량 분석 진행 중... ===")
            prompt_step2 = (
                "다음 짐 목록을 참고하여, 전체 짐이 차지할 총 공간과 효율적인 배치 방안을 분석해 주세요. "
                "짐을 겹치거나 쌓아서 공간을 최소화하는 방법을 포함하여 설명해주세요."
            )
            space_analysis_text = self._call_gemini_api(prompt_step2 + json.dumps(estimated_data, ensure_ascii=False, indent=2))
            print("✅ 2단계 성공: 총 공간 요구량 분석 완료.")
            print(f"분석 결과:\n{space_analysis_text}")

            # ===============================================
            # 3단계: 최종 배치 계획 수립 (JSON만 사용)
            # ===============================================
            print("\n=== [3단계] 최종 배치 계획 수립 진행 중... ===")
            prompt_step3 = (
                "다음 짐 목록과 3명의 탑승객을 고려하여, 가장 효율적인 차량 배치 계획을 제안해주세요."
                "트렁크 공간과 뒷좌석 활용 방안을 구체적으로 설명해주세요."
            )
            final_plan_text = self._call_gemini_api(prompt_step3 + json.dumps(estimated_data, ensure_ascii=False, indent=2))
            print("✅ 3단계 성공: 최종 배치 계획 수립 완료.")
            print(f"최종 계획:\n{final_plan_text}")
        
        except Exception as e:
            print(f"🚨 분석 중 치명적인 예외 발생: {e}")

        print("\n--- 분석 종료 ---")

if __name__ == "__main__":
    analyzer = GeminiMultiStepAnalyzer()
    # TODO: 아래 파일 경로를 사용자의 실제 파일 경로로 수정해야 합니다.
    analyzer.run_analysis(
        json_file_path="SW/Chain2/main_v3/scenarios/test2.txt", 
        image_file_path="SW/Chain2/main_v3/scenarios/test2.jpg"
    )