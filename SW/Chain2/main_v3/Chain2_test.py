import google.generativeai as genai
import json
import os
from PIL import Image

# 1. 여기에 YOUR_API_KEY를 본인의 실제 Gemini API 키로 변경하세요.
genai.configure(api_key="~~")

class GeminiTester:
    def __init__(self, model_name="gemini-2.5-flash"): # 모델 이름을 2.5-flash로 변경
        self.model = genai.GenerativeModel(model_name)

    def _ask_gemini(self, prompt, image=None):
        """
        Gemini API를 호출하고 결과를 반환합니다.
        오류가 발생하면 오류 메시지를 출력합니다.
        """
        try:
            content_parts = [prompt]
            if image:
                content_parts.append(image)

            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]

            response = self.model.generate_content(
                content_parts,
                safety_settings=safety_settings
            )

            if response.candidates and response.text:
                return f"✅ 성공: 응답이 정상적으로 생성되었습니다. (일부 생략: {response.text[:100]}...)"
            else:
                finish_reason = response.prompt_feedback.finish_reason.name if hasattr(response.prompt_feedback, 'finish_reason') else "UNKNOWN"
                return f"🚨 실패: 응답이 차단되었습니다. (reason: {finish_reason})"

        except Exception as e:
            return f"🚨 예외 발생: {e}"

    def run_tests(self, json_file_path, image_file_path):
        """
        단계별 테스트를 실행합니다.
        """
        print("--- 테스트 시작 ---")
        
        # 1단계: 텍스트(JSON)만으로 테스트
        print("\n=== [1단계] 텍스트(JSON)만으로 테스트 진행 중... ===")
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = f.read()
            prompt = f"다음 JSON 데이터를 분석하여 요약해주세요:\n{json_data}"
            result = self._ask_gemini(prompt)
            print(result)
        except Exception as e:
            print(f"🚨 텍스트 파일 로드 중 예외 발생: {e}")

        # 2단계: 이미지만으로 테스트
        print("\n=== [2단계] 이미지만으로 테스트 진행 중... ===")
        try:
            image = Image.open(image_file_path)
            prompt = "이 사진에 어떤 물건들이 있나요?"
            result = self._ask_gemini(prompt, image)
            print(result)
        except Exception as e:
            print(f"🚨 이미지 파일 로드 중 예외 발생: {e}")

        # 3단계: 이미지와 텍스트(JSON 단순화)로 테스트
        print("\n=== [3단계] 이미지와 단순화된 텍스트로 테스트 진행 중... ===")
        try:
            image = Image.open(image_file_path)
            with open(json_file_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            # JSON 데이터 키 불일치 수정
            simplified_data = {
                "people_count": original_data['people'],
                "total_luggage_count": original_data['total_luggage_count'],
                "luggage_details": original_data['luggage_details']
            }
            
            simplified_json = json.dumps(simplified_data, ensure_ascii=False, indent=2)
            prompt = f"이 사진과 다음 짐 목록을 참고하여 탑승객과 짐 배치를 제안해주세요:\n{simplified_json}"
            result = self._ask_gemini(prompt, image)
            print(result)
        except Exception as e:
            print(f"🚨 이미지/텍스트 처리 중 예외 발생: {e}")
        
        print("\n--- 테스트 종료 ---")

if __name__ == "__main__":
    tester = GeminiTester()
    # TODO: 아래 파일 경로를 사용자의 실제 파일 경로로 수정해야 합니다.
    tester.run_tests(
        json_file_path="SW/Chain2/main_v3/scenarios/test2.txt", 
        image_file_path="SW/Chain2/main_v3/scenarios/test2.jpg"
    )