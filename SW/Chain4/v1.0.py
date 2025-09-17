import json
import re
from typing import Dict, List, Union


class TaskSequenceConverter:
    """
    Task sequence를 16자리 십진수로 변환하는 클래스
    """
    
    def __init__(self):
        # 함수별 인코딩 규칙 정의
        self.encoding_rules = {
            'disk_rotate': {
                0: '0000',
                90: '0010'
            },
            'move_on_rail': {
                'M': '0000',
                'A': '0100', 
                'C': '0200'
            },
            'seat_rotate': {
                0: '0000',
                90: '1000',
                180: '2000',
                270: '3000'
            },
            'unfold': '0000',
            'fold': '0001'
        }
    
    def parse_function_call(self, func_call: str) -> Dict[str, Union[str, int]]:
        """
        함수 호출 문자열을 파싱하여 함수명과 매개변수를 추출
        
        Args:
            func_call: 함수 호출 문자열 (예: "disk_rotate(1, 90)")
            
        Returns:
            Dict: {'function': 함수명, 'id': id값, 'param': 매개변수}
        """
        # 정규표현식을 사용하여 함수명과 매개변수 추출
        pattern = r'(\w+)\((\d+)(?:,\s*(\w+))?\)'
        match = re.match(pattern, func_call.strip())
        
        if not match:
            raise ValueError(f"Invalid function call format: {func_call}")
        
        function_name = match.group(1)
        cell_id = int(match.group(2))
        param = match.group(3) if match.group(3) else None
        
        # 매개변수 타입 변환
        if param and param.isdigit():
            param = int(param)
        
        return {
            'function': function_name,
            'id': cell_id,
            'param': param
        }
    
    def encode_function(self, function_data: Dict[str, Union[str, int]]) -> str:
        """
        함수 데이터를 4자리 핀으로 인코딩
        
        Args:
            function_data: 파싱된 함수 데이터
            
        Returns:
            str: 4자리 핀 코드
        """
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
        """
        하나의 셀에 대한 함수 호출들을 처리하여 4자리 결과 생성
        
        Args:
            function_calls: 함수 호출 문자열 리스트
            
        Returns:
            str: 4자리 인코딩된 결과
        """
        total_sum = 0
        
        # 각 함수 호출을 인코딩하고 합계 계산
        for func_call in function_calls:
            parsed_func = self.parse_function_call(func_call)
            encoded_pin = self.encode_function(parsed_func)
            total_sum += int(encoded_pin)
        
        # 마지막 자리를 2로 나눈 몫으로 변환
        last_digit = total_sum % 10
        new_last_digit = last_digit // 2
        
        # 새로운 총합 계산
        final_result = (total_sum // 10) * 10 + new_last_digit
        
        # 4자리로 포맷팅 (앞에 0 채우기)
        return f"{final_result:04d}"
    
    def convert_to_16_digit(self, task_sequence: Dict[str, List[str]]) -> str:
        """
        전체 task_sequence를 16자리 십진수로 변환
        
        Args:
            task_sequence: JSON 형태의 task sequence 딕셔너리
            
        Returns:
            str: 16자리 십진수 문자열
        """
        result_parts = []
        
        # 1~4번 셀 순서대로 처리
        for cell_id in ['1', '2', '3', '4']:
            if cell_id in task_sequence:
                cell_result = self.process_cell(task_sequence[cell_id])
            else:
                # 해당 셀이 없으면 0000으로 설정
                cell_result = "0000"
            
            result_parts.append(cell_result)
        
        return ''.join(result_parts)
    
    def convert_from_json_string(self, json_string: str) -> str:
        """
        JSON 문자열로부터 직접 변환
        
        Args:
            json_string: JSON 형태의 문자열
            
        Returns:
            str: 16자리 십진수 문자열
        """
        try:
            data = json.loads(json_string)
            if 'task_sequence' in data:
                return self.convert_to_16_digit(data['task_sequence'])
            else:
                return self.convert_to_16_digit(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")


def main():
    """
    메인 실행 함수 - JSON 입력값 처리
    """
    print("=== Task Sequence to 16-Digit Decimal Converter ===")
    print("JSON 형식의 task_sequence 데이터를 입력해주세요.")
    print("예시 형식:")
    print('{"task_sequence": {"1": ["disk_rotate(1, 90)", "fold(1)"], ...}}')
    print("\n입력 (종료하려면 'quit' 입력):")
    
    converter = TaskSequenceConverter()
    
    while True:
        try:
            # 사용자로부터 JSON 입력 받기
            user_input = input("\nJSON 입력: ").strip()
            
            # 종료 조건
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("프로그램을 종료합니다.")
                break
            
            # 빈 입력 처리
            if not user_input:
                print("입력값이 비어있습니다. 다시 입력해주세요.")
                continue
            
            # JSON 파싱 및 변환
            data = json.loads(user_input)
            
            # task_sequence 키가 있는지 확인
            if 'task_sequence' in data:
                task_sequence = data['task_sequence']
            else:
                # task_sequence 키가 없으면 전체 데이터를 task_sequence로 간주
                task_sequence = data
            
            # 변환 실행
            result = converter.convert_to_16_digit(task_sequence)
            print(f"\n✅ 16자리 십진수 결과: {result}")
            
            # 상세 계산 과정 출력 여부 확인
            show_details = input("상세 계산 과정을 보시겠습니까? (y/n): ").strip().lower()
            
            if show_details in ['y', 'yes']:
                print("\n=== 셀별 상세 계산 과정 ===")
                for cell_id in ['1', '2', '3', '4']:
                    if cell_id in task_sequence:
                        functions = task_sequence[cell_id]
                        cell_result = converter.process_cell(functions)
                        print(f"\n📍 셀 {cell_id}: {cell_result}")
                        
                        # 상세 계산 과정 표시
                        total = 0
                        encoded_values = []
                        for func in functions:
                            parsed = converter.parse_function_call(func)
                            encoded = converter.encode_function(parsed)
                            total += int(encoded)
                            encoded_values.append(encoded)
                            print(f"   {func} → {encoded}")
                        
                        print(f"   합계: {' + '.join(encoded_values)} = {total}")
                        
                        last_digit = total % 10
                        new_last_digit = last_digit // 2
                        final = (total // 10) * 10 + new_last_digit
                        
                        print(f"   마지막 자리 변환: {last_digit} ÷ 2 = {new_last_digit}")
                        print(f"   최종 결과: {final:04d}")
                    else:
                        print(f"\n📍 셀 {cell_id}: 0000 (데이터 없음)")
        
        except json.JSONDecodeError as e:
            print(f"❌ JSON 형식 오류: {e}")
            print("올바른 JSON 형식으로 다시 입력해주세요.")
        
        except ValueError as e:
            print(f"❌ 값 오류: {e}")
        
        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
            break
        
        except Exception as e:
            print(f"❌ 예상치 못한 오류: {e}")


if __name__ == "__main__":
    main()