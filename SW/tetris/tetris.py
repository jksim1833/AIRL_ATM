# tetris.py

import sys
import argparse
from pathlib import Path
from time import perf_counter 
import time

HERE = Path(__file__).resolve().parent
MC_DIR = HERE / "main_chain"
UI_DIR = HERE / "user_input"

for p in (MC_DIR, UI_DIR):
    if not p.exists():
        raise FileNotFoundError(f"필수 폴더가 없습니다: {p}")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from user_input import get_user_input_web, get_user_input_scenario
import main_chain as MC

# rpi_controller 로드 
RPI_DIR = HERE / "rpi_controller"
RPI_FILE = RPI_DIR / "rpi_controller.py"
if not RPI_FILE.exists():
    raise FileNotFoundError(f"필수 파일이 없습니다: {RPI_FILE}")
if str(RPI_DIR) not in sys.path:
    sys.path.insert(0, str(RPI_DIR))
import rpi_controller as RPI

def run_pipeline(mode: str, port: int = 5002, open_browser: bool = True) -> dict:
    # 1) 입력 수집
    if mode == "web":
        people_count, image_data_url, scenario = get_user_input_web(
            port=port, auto_open_browser=open_browser
        )
    else:  # scenario 모드
        people_count, image_data_url, scenario = get_user_input_scenario()

    # 2) main_chain 입력 생성
    user_msgs = MC.make_chain1_user_input(
        people_count=people_count, image_data_url=image_data_url
    )

    # 2-1) 출력 파일 경로 준비
    OUT_ROOT = HERE / "tetris_out"
    OUT_DIR = OUT_ROOT / ("out_rt" if mode == "web" else "out_scenario")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{scenario}.txt"

    # 3) 전체 체인 실행 
    print("----- chain 실행중 -----")
    t_chain_start = perf_counter()
    try:
        result = MC.tetris_chain.invoke({"user_input": user_msgs, "people_count": people_count})
    except Exception as e:
        print("\n[ERROR] main_chain 실행 실패")
        print(f"- 이유: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
    t_chain_end = perf_counter()
    chain_elapsed = t_chain_end - t_chain_start

    # 3-1) chain4_out → 아두이노 전송 
    chain4_out = result["chain4_out"].strip()
    print("\n----- Arduino 제어 시작 -----")
    try:
        RPI.connect_to_arduinos()

        connected = getattr(RPI, "arduino_connections", {})
        if not connected:
            print("[WARN] 연결된 아두이노가 없습니다. DRY-RUN 모드로 진행합니다.")
            print(f"[DRY-RUN] 16-digit code: {chain4_out}")
        else:
            time.sleep(0.3)

            RPI.send_automated_command(chain4_out)

            time.sleep(2.0)

    except Exception as e:
        # 하드웨어 제어 중 예외가 나도 결과 저장은 계속 진행
        print(f"[WARN] 하드웨어 제어 중 예외 → DRY-RUN 전환: {e}")
        print(f"[DRY-RUN] 16-digit code: {chain4_out}")

    finally:
        # 연결 유무와 상관없이 안전 종료 시도
        try:
            RPI.close_all_connections()
        except Exception:
            pass
    print("----- Arduino 제어 종료 -----")

    # 4) 최종 출력 
    print("\n====================[ chain1_out ]====================")
    print(result.get("chain1_out", ""))
    print("\n====================[ chain2_out ]====================")
    print(result.get("chain2_out", ""))
    print("\n====================[ chain3_out ]====================")
    print(result.get("chain3_out", ""))
    print("\n====================[ chain4_out ]====================")
    print(chain4_out)

    # 5) 파일 저장 
    lines = []
    lines.append("====================[ chain1_out ]====================")
    lines.append(result.get("chain1_out", ""))
    lines.append("")
    lines.append("====================[ chain2_out ]====================")
    lines.append(result.get("chain2_out", ""))
    lines.append("")
    lines.append("====================[ chain3_out ]====================")
    lines.append(result.get("chain3_out", ""))
    lines.append("")
    lines.append("====================[ chain4_out ]====================")
    lines.append(chain4_out)

    out_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "out_path": out_path,
        "chain_elapsed": chain_elapsed,
    }


def main():
    ap = argparse.ArgumentParser(description="AI TETRIS launcher")
    ap.add_argument("--mode", required=True, choices=["web", "scenario"])
    ap.add_argument("--port", type=int, default=5002)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    # 전체 실행 시간 측정 시작
    t_total_start = perf_counter()
    res = run_pipeline(mode=args.mode, port=args.port, open_browser=(not args.no_browser))

    # 전체 실행 시간 측정 종료
    t_total_end = perf_counter()
    total_elapsed = t_total_end - t_total_start

    print("\n====================[ tetris 시스템 실행 완료 ]====================]")
    print(f"🕒 chain_run_time: {res['chain_elapsed']:.3f}s")
    print(f"🕒 tetris_run_time: {total_elapsed:.3f}s")

    with res["out_path"].open("a", encoding="utf-8") as f:
        f.write("\n====================[ tetris 시스템 실행 완료 ]====================]\n")
        f.write(f"🕒 chain_run_time: {res['chain_elapsed']:.3f}s\n")
        f.write(f"🕒 tetris_run_time: {total_elapsed:.3f}s\n")


if __name__ == "__main__":
    main()
