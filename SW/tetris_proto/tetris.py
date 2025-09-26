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

    # 3) 전체 체인 실행 (체인별 타이밍 & 즉시 출력)
    print("------------------------ chain 실행중 ------------------------------")
    t_chain_start = perf_counter()
    try:
        # chain1
        c1 = MC.run_chain1(user_msgs, people_count)
        print("\n=====================chain1_out =====================")
        print(c1["chain1_out"])
        print(f"\n🕒 chain1_run_time: {c1['chain1_run_time']:.3f}s")

        # chain2
        c2 = MC.run_chain2(user_msgs, c1["chain1_out"])
        print("\n=====================chain2_out =====================")
        print(c2["chain2_out"])
        print(f"\n🕒 chain2_run_time: {c2['chain2_run_time']:.3f}s")

        # chain3
        c3 = MC.run_chain3(c2["chain2_out"])
        print("\n=====================chain3_out =====================")
        print(c3["chain3_out"])
        print(f"\n🕒 chain3_run_time: {c3['chain3_run_time']:.3f}s")

        # chain4 (변환)
        c4 = MC.run_chain4(c3["chain3_out"])
        chain4_out = c4["chain4_out"].strip()
        print("\n=====================chain4_out =====================")
        print(chain4_out)

        # result 묶기
        result = {
            "chain1_out": c1["chain1_out"],
            "chain2_out": c2["chain2_out"],
            "chain3_out": c3["chain3_out"],
            "chain4_out": chain4_out,
            "chain1_run_time": c1["chain1_run_time"],
            "chain2_run_time": c2["chain2_run_time"],
            "chain3_run_time": c3["chain3_run_time"],
        }

    except Exception as e:
        print("\n[ERROR] main_chain 실행 실패")
        print(f"- 이유: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)

    t_chain_end = perf_counter()
    chain_elapsed = t_chain_end - t_chain_start

    # 3-1) chain4_out → 아두이노 전송
    print("\n----------------- Arduino 제어 시작 --------------------")
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
    print("-------------------- Arduino 제어 종료-------------------")

    # 4) 결과 파일 저장 (요구 포맷)
    lines = []
    lines.append("=====================chain1_out =====================")
    lines.append(result.get("chain1_out", ""))
    lines.append("")
    lines.append(f"🕒 chain1_run_time: {result['chain1_run_time']:.3f}s")
    lines.append("")
    lines.append("=====================chain2_out =====================")
    lines.append(result.get("chain2_out", ""))
    lines.append("")
    lines.append(f"🕒 chain2_run_time: {result['chain2_run_time']:.3f}s")
    lines.append("")
    lines.append("=====================chain3_out =====================")
    lines.append(result.get("chain3_out", ""))
    lines.append("")
    lines.append(f"🕒 chain3_run_time: {result['chain3_run_time']:.3f}s")
    lines.append("")
    lines.append("=====================chain4_out =====================")
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

    # 최종 요약(터미널)
    print("\n🕒 chain_run_time: {:.3f}s".format(res["chain_elapsed"]))
    print("🕒 tetris_run_time: {:.3f}s".format(total_elapsed))

    # 최종 요약(파일에 이어쓰기)
    with res["out_path"].open("a", encoding="utf-8") as f:
        f.write("\n🕒 chain_run_time: {:.3f}s\n".format(res["chain_elapsed"]))
        f.write("🕒 tetris_run_time: {:.3f}s\n".format(total_elapsed))


if __name__ == "__main__":
    main()
