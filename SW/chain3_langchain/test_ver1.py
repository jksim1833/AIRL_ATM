from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.callbacks import get_openai_callback
import tiktoken
import json
import os
import re
import argparse
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 토크나이저 설정
enc = tiktoken.get_encoding("cl100k_base")

# 디렉토리 설정 (langchain_v1 폴더 내의 경로)
dir_system = './langchain_v1/chain3_system'
dir_prompt = './langchain_v1/chain3_prompt'
dir_query = './langchain_v1/chain3_query'
prompt_load_order = ['chain3_prompt_role',
                     'chain3_prompt_function',
                     'chain3_prompt_environment',
                     'chain3_prompt_output_format',
                     'chain3_prompt_example']


class LangChainChatGPT:
    def __init__(self, prompt_load_order):
        # LangChain ChatOpenAI 모델 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            max_tokens=1000,
            top_p=0.5,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # JSON 출력 파서 설정
        self.output_parser = JsonOutputParser()
        
        self.messages = []
        self.max_token_length = 10000
        self.max_completion_length = 1000
        self.last_response = None
        self.query = ''
        
        # 시스템 프롬프트 로드
        fp_system = os.path.join(dir_system, 'chain3_system.txt')
        with open(fp_system, encoding='utf-8') as f:
            system_content = f.read()
        self.system_message = SystemMessage(content=system_content)
        
        # 프롬프트 파일들 로드
        for prompt_name in prompt_load_order:
            fp_prompt = os.path.join(dir_prompt, prompt_name + '.txt')
            with open(fp_prompt, encoding='utf-8') as f:
                data = f.read()
            
            # [user], [assistant] 태그로 분할
            data_split = re.split(r'\[user\]\n|\[assistant\]\n', data)
            data_split = [item for item in data_split if len(item) != 0]
            
            assert len(data_split) % 2 == 0
            for i, item in enumerate(data_split):
                if i % 2 == 0:
                    self.messages.append(HumanMessage(content=item))
                else:
                    self.messages.append(AIMessage(content=item))
        
        # 쿼리 파일 로드
        fp_query = os.path.join(dir_query, 'chain3_query.txt')
        with open(fp_query, encoding='utf-8') as f:
            self.query = f.read()
    
    def create_prompt_template(self):
        """ChatPromptTemplate 생성"""
        # 시스템 메시지와 대화 히스토리, 그리고 현재 질문을 포함한 템플릿
        template = ChatPromptTemplate.from_messages([
            self.system_message,
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}")
        ])
        return template
    
    def calculate_token_length(self, messages):
        """메시지들의 토큰 길이 계산"""
        total_content = ""
        for message in messages:
            if hasattr(message, 'content'):
                total_content += message.content
        return len(enc.encode(total_content))
    
    def truncate_messages_if_needed(self):
        """토큰 길이가 초과되면 오래된 메시지 제거"""
        current_length = self.calculate_token_length(self.messages)
        print(f'prompt length: {current_length}')
        
        while current_length > self.max_token_length - self.max_completion_length:
            if len(self.messages) >= 2:
                print('prompt too long. truncated.')
                # 가장 오래된 두 메시지 제거 (user-assistant 쌍)
                self.messages = self.messages[2:]
                current_length = self.calculate_token_length(self.messages)
            else:
                break
    
    def extract_json_part(self, text):
        """GPT 응답에서 JSON 부분만 추출"""
        if text.find('```') == -1:
            return text
        
        # 첫 번째와 두 번째 ``` 사이의 내용 추출
        start_idx = text.find('```') + 3
        end_idx = text.find('```', start_idx)
        if end_idx != -1:
            text_json = text[start_idx:end_idx]
        else:
            text_json = text[start_idx:]
        
        return text_json.strip()
    
    def generate(self, message, environment):
        """LangChain을 사용하여 응답 생성"""
        # 쿼리 템플릿에 환경과 명령어 삽입
        text_base = self.query
        if text_base.find('[ENVIRONMENT]') != -1:
            text_base = text_base.replace('[ENVIRONMENT]', json.dumps(environment))
        
        if text_base.find('[INSTRUCTION]') != -1:
            text_base = text_base.replace('[INSTRUCTION]', message)
        
        # 토큰 길이 확인 및 조정
        self.truncate_messages_if_needed()
        
        # 프롬프트 템플릿 생성
        prompt_template = self.create_prompt_template()
        
        # LLM 체인 생성
        chain = prompt_template | self.llm | self.output_parser
        
        try:
            # 토큰 사용량 추적과 함께 실행
            with get_openai_callback() as cb:
                # 응답 생성
                response = chain.invoke({
                    "chat_history": self.messages,
                    "query": text_base
                })
                
                print(f"Total Tokens: {cb.total_tokens}")
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Cost (USD): ${cb.total_cost}")
        
        except Exception as e:
            print(f"JSON 파싱 실패, 텍스트 응답으로 처리: {e}")
            # JSON 파싱 실패 시 텍스트 체인 사용
            text_chain = prompt_template | self.llm
            response_message = text_chain.invoke({
                "chat_history": self.messages,
                "query": text_base
            })
            
            text = response_message.content
            print(text)
            
            # JSON 부분 추출 및 처리
            self.last_response = self.extract_json_part(text)
            self.last_response = self.last_response.replace("'", "\"")
            
            # 응답을 파일로 저장
            with open('chain3_last_response.txt', 'w', encoding='utf-8') as f:
                f.write(self.last_response)
            
            try:
                response = json.loads(self.last_response, strict=False)
            except json.JSONDecodeError as json_error:
                print(f"JSON 파싱 실패: {json_error}")
                import pdb
                pdb.set_trace()
                return None
        
        # 응답 저장 및 처리
        self.last_response = json.dumps(response, ensure_ascii=False, indent=2)
        
        # 응답을 파일로 저장
        with open('chain3_last_response.txt', 'w', encoding='utf-8') as f:
            f.write(self.last_response)
        
        # 대화 히스토리에 추가
        self.messages.append(HumanMessage(content=text_base))
        self.messages.append(AIMessage(content=self.last_response))
        
        # 환경 업데이트
        if "environment_after" in response:
            self.environment = response["environment_after"]
        
        return response
    
    def dump_json(self, dump_name=None):
        """JSON 응답을 파일로 저장"""
        if dump_name is not None and self.last_response is not None:
            fp = os.path.join(dump_name + '.json')
            with open(fp, 'w', encoding='utf-8') as f:
                if isinstance(self.last_response, str):
                    # 문자열인 경우 그대로 저장
                    f.write(self.last_response)
                else:
                    # 딕셔너리인 경우 JSON으로 변환
                    json.dump(self.last_response, f, indent=4, ensure_ascii=False)


# 시나리오 선택 및 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scenario',
        type=str,
        required=True,
        help='scenario name (see the code for details)')
    args = parser.parse_args()
    scenario_name = args.scenario

    # 시나리오별 환경 및 명령어 설정
    if scenario_name == 'chain3_1seat':
        environment = {
            "objects": ["seat_1"],
            "object_states": {
                "seat_1": {
                    "type": "seat",
                    "position": [1, 1],
                    "direction": "A"
                }
            }
        }
        instructions = [
            "seat_1(storage, 1, 1, A)",
            "seat_1(storage, 1, 1, C)",
            "seat_1(seat, 2, 2, C)",
            "seat_1(storage, 0, 0, B)"
        ]
        
    elif scenario_name == 'chain3_4seat_compact':
        environment = {
            "objects": ["seat_1", "seat_2", "seat_3", "seat_4"],
            "object_states": {
                "seat_1": {"type": "seat", "position": [1, 1], "direction": "A"},
                "seat_2": {"type": "seat", "position": [1, 1], "direction": "A"},
                "seat_3": {"type": "seat", "position": [1, 1], "direction": "A"},
                "seat_4": {"type": "seat", "position": [1, 1], "direction": "A"}
            }
        }
        instructions = [
            "seat_1(storage, 0, 0, B), seat_2(storage, 2, 0, D), seat_3(storage, 0, 0, B), seat_4(storage, 2, 0, D)"
        ]
        
    elif scenario_name == 'chain3_4seat_pair_face':
        environment = {
            "objects": ["seat_1", "seat_2", "seat_3", "seat_4"],
            "object_states": {
                "seat_1": {"type": "storage", "position": [1, 0], "direction": "C"},
                "seat_2": {"type": "storage", "position": [1, 0], "direction": "C"},
                "seat_3": {"type": "storage", "position": [0, 1], "direction": "B"},
                "seat_4": {"type": "storage", "position": [2, 1], "direction": "D"}
            }
        }
        instructions = [
            "seat_1(seat, 1, 2, B), seat_2(seat, 1, 2, D), seat_3(seat, 1, 0, B), seat_4(seat, 1, 0, D)"
        ]
        
    else:
        parser.error('Invalid scenario name: ' + scenario_name)

    # AI 모델 초기화
    aimodel = LangChainChatGPT(prompt_load_order=prompt_load_order)

    # 출력 결과 저장 폴더 생성 (langchain_v1 폴더 내에)
    if not os.path.exists('./langchain_v1/chain3_out/' + scenario_name):
        os.makedirs('./langchain_v1/chain3_out/' + scenario_name)

    # 각 명령어에 대해 처리
    for i, instruction in enumerate(instructions):
        print(f"\n=== Instruction {i+1} ===")
        print(json.dumps(environment, indent=2, ensure_ascii=False))
        
        # GPT에게 instruction과 environment 전달하여 응답 생성
        response = aimodel.generate(instruction, environment)
        
        if response and "environment_after" in response:
            # 환경 상태 업데이트
            environment = response["environment_after"]
            
            # 응답을 JSON 파일로 저장 (langchain_v1 폴더 내에)
            aimodel.dump_json(f'./langchain_v1/chain3_out/{scenario_name}/{i}')
            
            print(f"✅ Step {i+1} completed successfully")
        else:
            print(f"❌ Step {i+1} failed")
            break

    print(f"\n🎉 Scenario '{scenario_name}' completed!")