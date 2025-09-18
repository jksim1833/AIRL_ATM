# 파일명: rpi_controller.py
# 설명: tetris.py로부터 16자리 자동화 명령을 받아 4개의 아두이노에 분배하여 전송하는 역할만 수행합니다.

import serial
import time
import serial.tools.list_ports

# --- 사용자 설정 영역 ---
# 제어할 아두이노의 시리얼 번호를 순서대로 입력하세요.
# 이 순서가 셀 1, 2, 3, 4번이 됩니다.
ARDUINO_SERIAL_NUMBERS = [
    '33437363436351303113',  # 셀 1번
    '3343736343635121F0B0',  # 셀 2번
    '33437363436351409183',  # 셀 3번
    '33437363436351010223'   # 셀 4번
]
BAUD_RATE = 9600
# 아두이노 펌웨어에서 setup() 완료 후 시리얼로 출력하는 메시지
# 이 메시지를 받아야 연결이 성공한 것으로 간주합니다.
ARDUINO_READY_MESSAGE = "Arduino Automation FW v2.1 Ready."
# ---------------------

# 아두이노 연결 관리 딕셔너리
arduino_connections = {}

def connect_to_arduinos():
    """
    [개선됨] 시리얼 번호로 아두이노를 찾고, 'Ready' 메시지를 수신하여
    통신 준비가 완료되었는지 확인한 후 연결합니다.
    """
    print("🔎 아두이노 연결을 시작합니다...")
    found_ports = list(serial.tools.list_ports.comports())
    
    for i, sn in enumerate(ARDUINO_SERIAL_NUMBERS):
        port_found = False
        for p in found_ports:
            if p.serial_number and p.serial_number == sn:
                try:
                    # 포트 연결 시도 (timeout을 넉넉하게 2초로 설정)
                    ser = serial.Serial(p.device, BAUD_RATE, timeout=2)
                    # 아두이노가 재부팅되고 시리얼을 초기화할 시간을 줍니다.
                    time.sleep(2)
                    
                    print(f"  - 셀 {i+1}번({p.device}) 응답 대기 중...")
                    
                    # 아두이노로부터 Ready 메시지가 오는지 최대 3초간 기다립니다.
                    ready = False
                    start_time = time.time()
                    while time.time() - start_time < 3:
                        response = ser.readline().decode('utf-8').strip()
                        if ARDUINO_READY_MESSAGE in response:
                            arduino_connections[sn] = ser
                            print(f"  ✅ 셀 {i+1}번 연결 성공 (SN: {sn}, Port: {p.device})")
                            ready = True
                            break # 응답 확인 후 루프 탈출
                    
                    if not ready:
                        print(f"  ⚠️  경고: 셀 {i+1}번에서 준비 완료 응답을 받지 못했습니다.")
                        ser.close()

                except serial.SerialException as e:
                    print(f"  ❌ 에러: 셀 {i+1}번 연결 실패 (SN: {sn}). {e}")
                finally:
                    port_found = True
                    break # 해당 시리얼 번호의 포트를 찾았으므로 다음 번호로 넘어감
        
        if not port_found:
            print(f"  ⚠️  경고: 셀 {i+1}번 아두이노를 찾을 수 없습니다 (SN: {sn}).")

def send_automated_command(full_command: str):
    """
    16자리 명령어를 4자리씩 나누어 각 아두이노에 전송합니다.
    """
    if not isinstance(full_command, str) or len(full_command) != 16 or not full_command.isdigit():
        print(f"❌ 잘못된 자동화 명령입니다: '{full_command}'. 16자리 숫자여야 합니다.")
        return

    print(f"\n⚙️  자동화 명령 실행: {full_command}")
    command_chunks = [full_command[i:i+4] for i in range(0, 16, 4)]
    
    for i, serial_num in enumerate(ARDUINO_SERIAL_NUMBERS):
        command_to_send = command_chunks[i]
        if serial_num in arduino_connections:
            try:
                conn = arduino_connections[serial_num]
                # 반드시 '\n'을 포함하여 전송해야 아두이노가 명령의 끝으로 인식합니다.
                conn.write((command_to_send + '\n').encode('utf-8'))
                print(f"  > 셀 {i+1}번에 전송: '{command_to_send}'")
            except Exception as e:
                 print(f"  > ❌ 에러: 셀 {i+1}번 전송 실패. {e}")
        else:
            print(f"  > 셀 {i+1}번 건너뛰기 (연결 안됨).")
    print("-" * 20)

def close_all_connections():
    """
    모든 아두이노를 멈추고 시리얼 포트를 닫습니다.
    """
    print("\n🔌 모든 시리얼 포트를 닫습니다...")
    
    # 종료 전 모든 모터를 정지시키는 '0000' 명령을 보냅니다.
    for i, serial_num in enumerate(ARDUINO_SERIAL_NUMBERS):
        if serial_num in arduino_connections:
            try:
                arduino_connections[serial_num].write(b'0000\n')
            except:
                pass # 에러가 나도 무시하고 다음 연결을 닫습니다.
    
    time.sleep(0.1) # 명령이 전송될 시간을 잠시 줍니다.

    for conn in arduino_connections.values():
        if conn and conn.isOpen():
            conn.close()
    
    # 딕셔너리 비우기
    arduino_connections.clear()
    print("✅ 모든 연결이 안전하게 종료되었습니다.")