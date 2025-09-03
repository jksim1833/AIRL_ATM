from openai import OpenAI
import tiktoken
import json
import os
import re
import argparse

enc = tiktoken.get_encoding("cl100k_base")   # 토크나이저 불러오는 코드 : 토큰 수 계산하기 위해. *토큰: 지피티가 텍스트를 나누어 처리하는 최소 단위 
with open('../tetris_secrets.json') as f:
    credentials = json.load(f)

dir_system = './chain3_system'
dir_prompt = './chain3_prompt'
dir_query = './chain3_query'
prompt_load_order = ['chain3_prompt_role',
                     'chain3_prompt_environment',
                     'chain3_prompt_function',
                     'chain3_prompt_output_format',
                     'chain3_prompt_example']


class ChatGPT:
    def __init__(
            self,
            credentials,
            prompt_load_order):
        self.client = OpenAI(api_key=credentials["openai"]["OPENAI_API_KEY"])

        self.credentials = credentials    #tetris_secrets.json에서 불러온 인증 정보 
        self.messages = []           #GPT에게 전달할 대화 내용을 리스트로 저장할 공간 
        self.max_token_length = 10000       #GPT에게 보낼 전체 프롬프트 최대 토큰 수 제한 
        self.max_completion_length = 8000         #GPT가 생성할 응답 최대 길이 
        self.last_response = None          #GPT가 마지막에 출력한 응답 저장용 변수 
        self.last_response_raw = None    #GPT가 마지막에 출력한 응답 저장용 변수 (raw)
        self.query = ''

        # load prompt file 시스템 프롬프트 읽기 
        fp_system = os.path.join(dir_system, 'chain3_system.txt')
        with open(fp_system, encoding='utf-8') as f:
            data = f.read()
        self.system_message = {"role": "system", "content": data}

        # load prompt file 프롬프트 읽기 
        for prompt_name in prompt_load_order:
            fp_prompt = os.path.join(dir_prompt, prompt_name + '.txt')
            with open(fp_prompt, encoding='utf-8') as f:
                data = f.read()
            data_spilit = re.split(r'\[user\]\n|\[assistant\]\n', data)
            data_spilit = [item for item in data_spilit if len(item) != 0]
            
            assert len(data_spilit) % 2 == 0
            for i, item in enumerate(data_spilit):
                if i % 2 == 0:
                    self.messages.append({"sender": "user", "text": item})
                else:
                    self.messages.append({"sender": "assistant", "text": item})

        fp_query = os.path.join(dir_query, 'chain3_query.txt')
        with open(fp_query, encoding='utf-8') as f:
            self.query = f.read()


    def create_prompt(self):
        prompt = [] #최종적으로 gpt에게 전달하는 메세지 리스트 
        prompt.append(self.system_message)
        
        for message in self.messages:
            prompt.append(
                {"role": message['sender'], "content": message['text']})
            
        prompt_content = ""   # 전체 텍스트 이어붙여 토큰 길이 계산용 변수

        for message in prompt:
            prompt_content += message["content"]
        print('prompt length: ' + str(len(enc.encode(prompt_content)))) # 현재 프롬프트의 토큰 길이 출력 (디버깅용)

        if len(enc.encode(prompt_content)) > self.max_token_length - \
                self.max_completion_length:
            print('prompt too long. truncated.')

            # truncate the prompt by removing the oldest two messages
            self.messages = self.messages[2:]
            prompt = self.create_prompt() 
        return prompt 

    # GPT 응답 중 json파트만 추출
    # json part is between ``` and ```.
    def extract_json_part(self, text):

        if text.find('```') == -1:
            return text
        
        text_json = text[text.find(
            '```') + 3:text.find('```', text.find('```') + 3)]
        return text_json
    
    def generate(self, message, environment, feedback = False):
        
        if feedback:
            self.messages.append({'sender': 'user', 'text': message})
        
        else:
            text_base = self.query
            if text_base.find('[ENVIRONMENT]') != -1:
                text_base = text_base.replace('[ENVIRONMENT]', json.dumps(environment))
            
            if text_base.find('[INSTRUCTION]') != -1:
                text_base = text_base.replace('[INSTRUCTION]', message)
                
            self.messages.append({'sender': 'user', 'text': text_base})

            
        response = self.client.chat.completions.create(
                model="gpt-4o",
                
                messages=self.create_prompt(),
                temperature=0.1,
                max_tokens =self.max_completion_length, 
                top_p=0.5,
                frequency_penalty=0.0,
                presence_penalty=0.0
        )
        text = response.choices[0].message.content
        print(text)

        self.last_response_raw = text #파싱 전 원문 저장(메모리)
        self.last_response = text
        self.last_response = self.extract_json_part(self.last_response)
        self.last_response = self.last_response.replace("'", "\"")
            
        # GPT응답을 텍스트 파일로 저장
        with open('chain3_last_response.txt', 'w') as f:
                f.write(self.last_response)

        try:
            self.json_dict = json.loads(self.last_response, strict=False)
            self.environment = self.json_dict["environment_after"]
        except BaseException:
            self.json_dict = None
            return None

        if len(self.messages) > 0 and self.last_response is not None:
            self.messages.append({"sender": "assistant", "text": self.last_response})

        return self.json_dict

    def dump_json(self, dump_name=None):
        if dump_name is not None:
            # dump the dictionary to json file dump 1, 2, ...
            fp = os.path.join(dump_name + '.json')
            with open(fp, 'w') as f:
                json.dump(self.json_dict, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument(
        '--scenario', #어떤 시나리오 실행할지 지정하는 옵션 
        type=str,
        required=True,
        help='scenario name (see the code for details)')
    args = parser.parse_args()
    scenario_name = args.scenario

    # 시나리오 파일에서 읽기
    scenario_path = os.path.join("chain3_scenario", f"{scenario_name}.json")
    if not os.path.exists(scenario_path):
        parser.error(f"scenario file not found: {scenario_path}")

    with open(scenario_path, "r") as f:
        scenario_data = json.load(f)

    environment = scenario_data["environment"]
    instructions = scenario_data["instructions"]
    
    # 시나리오 실행: GPT 호출 > 결과 저장 
    aimodel = ChatGPT(
        credentials,
        prompt_load_order=prompt_load_order)

    # 출력 결과 저장하는 폴더 생성 
    if not os.path.exists('./chain3_out/' + scenario_name): 
        os.makedirs('./chain3_out/' + scenario_name)

    max_trial = 3

    # 현재 환경 출력 
    for i, instruction in enumerate(instructions):
        print(json.dumps(environment)) 