import os

# 디렉토리 생성
os.makedirs('chain3_system', exist_ok=True)
os.makedirs('chain3_prompt', exist_ok=True)
os.makedirs('chain3_query', exist_ok=True)
os.makedirs('chain3_out', exist_ok=True)

# chain3_system.txt
with open('chain3_system/chain3_system.txt', 'w', encoding='utf-8') as f:
    f.write("""You are an intelligent seat control agent for a smart vehicle system called AI TETRIS. Given a seat arrangement plan and environment information, you break it down into a sequence of seat control actions.

""")

# chain3_prompt_role.txt
with open('chain3_prompt/chain3_prompt_role.txt', 'w', encoding='utf-8') as f:
    f.write("""[user]
You are an intelligent seat control agent for a smart vehicle system called AI TETRIS. Given a seat arrangement plan and environment information, you break it down into a sequence of seat control actions.
Please do not begin working until I say "Start working." Instead, simply output the message "Waiting for next input." Understood?
[assistant]
Understood. Waiting for next input.""")

# chain3_prompt_function.txt
with open('chain3_prompt/chain3_prompt_function.txt', 'w', encoding='utf-8') as f:
    f.write("""[user]
Necessary and sufficient seat control actions are defined as follows:

\"\"\"
"SEAT CONTROL LIST"

#Motion action
- move(seat_id, x, y): Move the specified seat to the given (x, y) position in the vehicle cabin layout. This action can be performed regardless of the seat's current state, whether "chair" (unfolded) or "storage" (folded). Avoid using unnecessary fold or unfold commands to ensure an efficient sequence.
- rotate(seat_id, degree): Rotate the seat by the specified degree (only allowed values: 0, 90, 180, or 270; do not use -180 or other values). The seat's current angle is considered 0 degrees. If the current state is "chair", fold(seat_id) must be executed beforehand. However, if the seat is already in the "storage" state, fold(seat_id) should be omitted.

#State action
- flip(seat_id): This action transforms the seat from a folded state into a chair-like configuration. It can only be used when the seat is currently folded. If used, this command must appear at the final stage of the seat arrangement sequence, after all move(seat_id, x, y) and rotate(seat_id, degree) commands have been completed.
- fold(seat_id): This action folds the entire seat into a compact, flat configuration to free up cabin space.
\"\"\"
-------------------------------------------------------
The texts above are part of the overall instruction. Do not start working yet:
[assistant]
Understood. Waiting for next input.""")

# chain3_prompt_environment.txt
with open('chain3_prompt/chain3_prompt_environment.txt', 'w', encoding='utf-8') as f:
    f.write("""[user]
Information about environments and objects are given as python dictionary. Example:
\"\"\"
{"environment":{
	"objects": ["seat_1"],
    	"object_states": {"seat_1": { "type": "chair", "position": [1, 1], "direction": "A"}}}}
\"\"\"
Object states are represented using those state sets:
\"\"\"
"STATE LIST"
- type: "chair" or "storage". physical form of the seat. 

- position: [x, y] coordinate of the seat. x, y must be integers in the range 0 ≤ x, y ≤ 2.
Each seat (seat_1 to seat_4) is controlled independently in separate physical regions. Even if their coordinates overlap, there is no physical conflict between them. 

- direction: "A", "B", "C", or "D". These represent four absolute orientations in the cabin.
-------------------------------------------------------
The texts above are part of the overall instruction. Do not start working yet:
[assistant]
Understood. I will wait for further instructions before starting to work.


""")

# chain3_prompt_output_format.txt
with open('chain3_prompt/chain3_prompt_output_format.txt', 'w', encoding='utf-8') as f:
    f.write("""[user]
You divide the actions given in the text into detailed seat control actions and put them together as a python dictionary.
The dictionary has four keys.
\"\"\"
- dictionary["environment_before"]: The state of the environment before the manipulation.
- dictionary["instruction"]: description of the seat arrangement.
- dictionary["task_cohesion"]: A dictionary containing information about the seat's actions that have been split up.
- dictionary["environment_after"]: The state of the environment after the manipulation.
\"\"\"
Two keys exist in dictionary["task_cohesion"].
\"\"\"
- dictionary["task_cohesion"]["task_sequence"]: Contains a list of seat control actions. Only the behaviors defined in the "SEAT CONTROL LIST" will be used.
- dictionary["task_cohesion"]["step_instructions"]: contains a list of instructions corresponding to dictionary["task_cohesion"]["task_sequence"].
\"\"\"
Always place the "instruction" key at the top of the output dictionary.
-------------------------------------------------------
The texts above are part of the overall instruction. Do not start working yet:
[assistant]
Understood. Waiting for next input.""")

# chain3_prompt_example.txt
with open('chain3_prompt/chain3_prompt_example.txt', 'w', encoding='utf-8') as f:
    f.write("""[user]
I will give you some examples of the input and the output you will generate. 
Example 1:
\"\"\"
- Input:
{"objects": ["seat_1"],
"object_states": {"seat_1": {"type": "chair", "position": [1, 1], "direction": "A"}},
"instruction": "seat_1(chair, 0, 1, B)"}
- Output:
```
{"environment_before": {
    "objects": ["seat_1"],
    "object_states": {"seat_1": {"type": "chair", "position": [1, 1], "direction": "C"}}},
"instruction": "seat_1(chair, 0, 1, B)",
"task_cohesion": {
    "task_sequence": [
       "fold(seat_1)",
       "rotate(seat_1, 90)",
       "move(seat_1, 0, 1)",
       "flip(seat_1)"
    ],
    "step_instructions": [
       "fold the seat_1 to convert it into storage mode",
       "rotate the seat_1 90 degrees clockwise",
       "move the seat_1 to position (0, 1)",
       "flip the seat_1 to convert it into chair mode"
    ]},
"environment_after": {
    "objects": ["seat_1"],
    "object_states": {"seat_1": {"type": "chair", "position": [0, 1], "direction": "B"}}}}
```
\"\"\"
Example 2:
\"\"\"
- Input:
{"objects": ["seat_1"],
"object_states": {"seat_1": {"type": "chair", "position": [1, 1], "direction": "C"}},
"instruction": "seat_1(storage, 0, 0, D)"}
- Output:
```
{"environment_before": {
    "objects": ["seat_1"],
    "object_states": {"seat_1": {"type": "chair", "position": [1, 1], "direction": "C"}}},
"instruction": "seat_1(storage, 0, 0, D)",
"task_cohesion": {
    "task_sequence": [
      "fold(seat_1)",
      "rotate(seat_1, 90)",
      "move(seat_1, 0, 0)"
    ],
    "step_instructions": [
      "fold the seat_1 to convert it into storage mode",
      "rotate the seat_1 90 degrees clockwise",
      "move the seat_1 to position (0, 0)"
    ]},
"environment_after": {
    "objects": ["seat_1"],
    "object_states": {"seat_1": {"type": "storage", "position": [0, 0], "direction": "D"}}}}
```
\"\"\"
From these examples, learn that some seat control actions have dependencies with the actions before and after them.
-------------------------------------------------------
The texts above are part of the overall instruction. Do not start working yet:
[assistant]
Understood. I will wait for further instructions before starting to work.""")

# chain3_query.txt
with open('chain3_query/chain3_query.txt', 'w', encoding='utf-8') as f:
    f.write("""Start working. Resume from the environment below.
\"\"\"
{"environment":[ENVIRONMENT]}
\"\"\"
The instruction is as follows:
\"\"\"
{"instruction": [INSTRUCTION]}
\"\"\"
The dictionary that you return should be formatted as valid JSON object (not Python dict), with all keys and string values using double quotes. Follow these rules:
1. Make sure that each element of the ["step_instructions"] explains corresponding element of the ["task_sequence"]. Refer to the "SEAT CONTROL LIST" to understand the elements of ["task_sequence"].
2. The length of the ["step_instructions"] list must be the same as the length of the ["task_sequence"] list.
3. Never left ',' at the end of the list.
4. Keep track of all items listed in the "objects" section of the "environment_before" field. Please ensure that you fill out both the "objects" and "object_states" sections for all listed items. 
5. All keys of the dictionary should be double-quoted.
6. Insert ``` at the beginning and the end of the dictionary to separate it from the rest of your response, with no other text before or after. Do not include the word "json" after the opening backticks. Only use plain backticks (```).
7. your seat control sequence must obey the following constraints:

- move(seat_id, x, y) can only be used when the seat is in the "storage" state.
- rotate(seat_id, degree) must be executed before flip(seat_id).
- flip(seat_id) must appear at the final stage of the sequence.
- If the seat's direction is the same in both environment_before and environment_after, do NOT generate a rotate command in the seat control actions.
- if "direction" changes from A to B or B to C or C to D or D to A, "degree" is 90
- if "direction" changes from A to D or D to C or C to B or B to A, "degree" is 270
8. You must output strictly in the same format as the examples in "chain3_prompt_example".

Adhere to the output format I defined above. Follow the eight rules. Think step by step.""")

# .env 파일 생성
with open('.env', 'w', encoding='utf-8') as f:
    f.write("""OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_TRACING_V2=false
""")

print("✅ 모든 파일과 디렉토리가 생성되었습니다!")
print("📝 .env 파일을 편집하여 실제 OpenAI API 키를 입력하세요.")
print("🚀 준비 완료! 이제 다음 명령어로 실행할 수 있습니다:")
print("   python test_ver1.py --scenario chain3_1seat")