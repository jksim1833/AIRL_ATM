# tetris.py

import os
import sys
import argparse
from pathlib import Path
from time import perf_counter 
import time
import json  # [FIX] json 사용부 대비
from dotenv import load_dotenv
load_dotenv(override=True)

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

def run_pipeline(
    mode: str,
    port: int = 5002,
    open_browser: bool = True,
    *,
    override: dict | None = None,                   # [EVAL] 입력 오버라이드
    langsmith_tags: list[str] | None = None,        # [EVAL] LangSmith tags
    langsmith_metadata: dict | None = None,         # [EVAL] LangSmith metadata
    eval_dry_run_hw: bool = False,                  # [EVAL] 하드웨어 전송 생략
) -> dict:
    # 1) 입력 수집
    if override and {"people_count","image_data_url","scenario"} <= set(override.keys()):   # [EVAL]
        people_count = int(override["people_count"])
        image_data_url = str(override["image_data_url"])
        scenario = str(override["scenario"])
    else:
        if mode == "web":
            people_count, image_data_url, scenario = get_user_input_web(
                port=port, auto_open_browser=open_browser
            )
        else:
            people_count, image_data_url, scenario = get_user_input_scenario()

    # 2) main_chain 입력 생성 (원문 유지)
    user_msgs = MC.make_chain1_user_input(
        people_count=people_count, image_data_url=image_data_url
    )

    # 2-1) (추적 로그용) LangSmith config 구성: ON일 때만 전달
    invoke_cfg = None  # [FIX] OFF일 때는 None 전달
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        invoke_cfg = {
            "tags": (langsmith_tags or []),
            "metadata": (langsmith_metadata or {
                "scenario": scenario,
                "people": people_count,
                "mode": mode,
            }),
        }

    # 3) 체인별 순차 실행 + 즉시 출력
    print("---------------------------- chain 실행중 --------------------------------")
    lines: list[str] = []

    def _append_header(title: str):
        lines.append(f"================================[ {title} ]==================================")

    total_chain_time = 0.0

    # Chain1
    c1 = MC.run_chain1(user_msgs=user_msgs, people_count=people_count, config=invoke_cfg)
    chain1_out, t1 = c1["chain1_out"], c1["elapsed"]; total_chain_time += t1
    print("\n====================[ chain1_out ]====================")

    try:
        pretty_c1 = json.dumps(json.loads(chain1_out), ensure_ascii=False, indent=2)  # [FIX]
        print(pretty_c1)
    except Exception:
        pretty_c1 = chain1_out  # [FIX] 저장본에도 동일 반영
        print(chain1_out)
    print(f"\n🕒 [chain1_out 생성 시간]: {t1:.3f}초")

    # [FIX] 저장 파일에도 체계적으로 포함
    lines.append("====================[ chain1_out ]====================")
    lines.append(pretty_c1)
    lines.append(f"🕒 [chain1_out 생성 시간]: {t1:.3f}초")

    # Chain2
    c2 = MC.run_chain2(user_msgs=user_msgs, chain1_out=chain1_out, config=invoke_cfg)
    chain2_out, t2 = c2["chain2_out"], c2["elapsed"]; total_chain_time += t2
    print("\n====================[ chain2_out ]====================")
    print(chain2_out)
    print(f"\n🕒 [chain2_out 생성 시간]: {t2:.3f}초")
    _append_header("chain2_out"); lines.append(chain2_out)
    lines.append(f"🕒 [chain2_out 생성 시간]: {t2:.3f}초")

    # Chain3
    c3 = MC.run_chain3(chain2_out=chain2_out, config=invoke_cfg)
    chain3_out, t3 = c3["chain3_out"], c3["elapsed"]; total_chain_time += t3
    print("\n====================[ chain3_out ]====================")
    print(chain3_out)
    print(f"\n🕒 [chain3_out 생성 시간]: {t3:.3f}초")
    _append_header("chain3_out"); lines.append(chain3_out)
    lines.append(f"🕒 [chain3_out 생성 시간]: {t3:.3f}초")

    # Chain4
    try:
        c4 = MC.run_chain4(chain3_out=chain3_out, config=invoke_cfg)  # config는 시그니처 통일용
        chain4_out, t4 = c4["chain4_out"], c4["elapsed"]; total_chain_time += t4
    except Exception as e:
        print(f"[ERROR] chain4 변환 실패: {e}")
        chain4_out, t4 = "", 0.0
    print("\n====================[ chain4_out ]====================")
    print(chain4_out)
    print(f"\n🕒 [chain4_out 생성 시간]: {t4:.3f}초")
    _append_header("chain4_out"); lines.append(chain4_out)
    lines.append(f"🕒 [chain4_out 생성 시간]: {t4:.3f}초")

    chain_run_time = total_chain_time

    # 3-1) chain4_out → 아두이노 전송
    print("\n----------------------- Arduino 제어 시작 ---------------------------")
    try:
        if eval_dry_run_hw:
            print(f"[EVAL-DRY-RUN] 하드웨어 전송 생략. 16-digit: {chain4_out}")
        else:
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
        print(f"[WARN] 하드웨어 제어 중 예외 → DRY-RUN 전환: {e}")
        print(f"[DRY-RUN] 16-digit code: {chain4_out}")
    finally:
        try:
            RPI.close_all_connections()
        except Exception:
            pass
    print("-------------------------- Arduino 제어 종료 ---------------------------")

    print(f"\n🕒 [chain_run_time]: {chain_run_time:.3f}s")

    # 5) 파일 저장 (운영모드 + 추적로그 OFF일 때만 저장)
    out_path = None
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() != "true":
        OUT_DIR = (HERE / "tetris_out" / ("out_web" if mode == "web" else "out_scenario"))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = OUT_DIR / f"{scenario}_{ts}.txt"
        out_text = "\n".join(lines + [f"\n🕒 [chain_run_time]: {chain_run_time:.3f}s"])
        out_path.write_text(out_text, encoding="utf-8")

    return {
        "out_path": out_path,
        "chain_elapsed": chain_run_time,
        "chain_run_time": chain_run_time,
        "scenario": scenario,
        "people_count": people_count,
        "chain1_out": chain1_out,
        "chain2_out": chain2_out,
        "chain3_out": chain3_out,
        "chain4_out": chain4_out,
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

    print("\n====================[ tetris 시스템 실행 완료 ]====================")  # [FIX]
    print(f"🕒 [chain_run_time]: {res.get('chain_run_time', res['chain_elapsed']):.3f}s")
    print(f"🕒 [tetris_run_time]: {total_elapsed:.3f}s")

    # 파일은 운영 + 추적로그 OFF일 때만 저장하므로, 존재할 때만 최종 2줄 append
    if res.get("out_path"):
        with res["out_path"].open("a", encoding="utf-8") as f:
            f.write("\n====================[ tetris 시스템 실행 완료 ]====================\n")  # [FIX]
            f.write(f"🕒 [chain_run_time]: {res.get('chain_run_time', res['chain_elapsed']):.3f}s\n")
            f.write(f"🕒 [tetris_run_time]: {total_elapsed:.3f}s\n")


if __name__ == "__main__":
    main()
