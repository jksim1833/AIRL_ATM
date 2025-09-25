# tetris.py

import sys
import argparse
from pathlib import Path
from time import perf_counter
import time
import os, json, re

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

# ===== LangSmith Feedback(지표) 관련 =====
from langsmith import Client  # pip install langsmith

AS_ROOT = HERE / "answer_sheet"  # 정답셋 루트: answer_sheet/chain1|chain2|chain3/...

def _norm_json(s: str) -> dict:
    s = (s or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S | re.I)
    if m: s = m.group(1).strip()
    if not (s.startswith("{") and s.endswith("}")):
        first = s.find("{"); last = s.rfind("}")
        if first != -1 and last != -1 and first < last:
            s = s[first:last+1]
    return json.loads(s)

def _eval_chain1_count(chain1_out_text: str, gold_count: int) -> dict:
    try:
        d = json.loads(chain1_out_text)
        pred = int(d.get("total_luggage_count", -999))
        acc = 1.0 if pred == gold_count else 0.0
        return {"success": 1.0, "count_pred": pred, "count_accuracy": acc}
    except Exception as e:
        return {"success": 0.0, "error": f"parse-error: {e}"}

def _canon(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _eval_chain2_accuracy(chain2_out_text: str, gold_obj: dict) -> dict:
    try:
        pred = _norm_json(chain2_out_text)
        acc = 1.0 if _canon(pred) == _canon(gold_obj) else 0.0
        return {"success": 1.0, "accuracy": acc}
    except Exception as e:
        return {"success": 0.0, "error": f"parse-error: {e}"}

def _normalize_seq_tokens_from_obj(obj: dict) -> set[str]:
    # obj가 {"task_sequence": {...}} 이거나 곧바로 {...} 여도 처리
    seq = obj.get("task_sequence", obj)
    toks = set()
    for seat, actions in seq.items():
        if isinstance(actions, str):
            actions = [actions]
        for a in actions or []:
            tok = re.sub(r"\s+", "", str(a))
            toks.add(f"{seat}:{tok}")
    return toks

def _eval_chain3_f1(chain3_out_text: str, gold_obj: dict) -> dict:
    try:
        pred = _norm_json(chain3_out_text)
        P = _normalize_seq_tokens_from_obj(pred)
        G = _normalize_seq_tokens_from_obj(gold_obj)
        tp = len(P & G); fp = len(P - G); fn = len(G - P)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        return {"success": 1.0, "precision": prec, "recall": rec, "f1": f1}
    except Exception as e:
        return {"success": 0.0, "error": f"parse-error: {e}"}

def _push_feedback(client: Client, project: str, tags: list[str], k: str, score: float, comment: str | None = None, metadata: dict | None = None):
    runs = client.list_runs(project_name=project, filter={"tags": tags}, limit=1, order="desc")
    run = next(iter(runs), None)
    if run:
        client.create_feedback(run.id, key=k, score=score, comment=comment, metadata=metadata or {})

def _load_chain3_gold(scenario: str, people_count: int) -> dict | None:

    p = AS_ROOT / "chain3" / scenario / f"people_{people_count}" / f"{scenario}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None

def run_pipeline(mode: str, port: int = 5002, open_browser: bool = True,
                 run_type: str = "operate", trace_on: bool | None = None,
                 model_override: str | None = None, temp_override: float | None = None) -> dict:
    # === 정책 강제: 평가모드는 scenario 전용 + tracing ON ===
    if run_type == "eval":
        if mode != "scenario":
            raise ValueError("평가모드는 scenario 모드에서만 가능합니다.")
        trace_on = True  # 강제

    # 0) 트레이싱 on/off
    if trace_on is True:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    elif trace_on is False:
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
    # trace_on is None → 운영모드에서 .env 그대로 유지

    # 0-1) 모델/온도 오버라이드
    if model_override: os.environ["TETRIS_LLM_MODEL"] = model_override
    if temp_override is not None: os.environ["TETRIS_LLM_TEMPERATURE"] = str(temp_override)

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

    # 2-1) 출력 파일 경로 준비(로컬 저장 유지)
    OUT_ROOT = HERE / "tetris_out"
    OUT_DIR = OUT_ROOT / ("out_rt" if mode == "web" else "out_scenario")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{scenario}.txt"

    # 2-2) 공통 태그 (people 태그 추가!)
    model_name = os.getenv("TETRIS_LLM_MODEL", "gemini-2.5-flash")
    temp_val   = float(os.getenv("TETRIS_LLM_TEMPERATURE", "0.2"))
    common_tags = [
        "tetris",
        f"scenario:{scenario}",
        f"people:{int(people_count)}",
        f"model:{model_name}",
        f"temp:{temp_val}",
        f"run:{run_type}",
    ]

    # 3) 전체 체인 실행 
    print("----- chain 실행중 -----")
    t_chain_start = perf_counter()
    try:
        result = MC.tetris_chain_with(meta_tags=common_tags).invoke(
            {"user_input": user_msgs, "people_count": people_count},
            config={"tags": common_tags, "metadata": {"scenario": scenario, "people": int(people_count), "mode": mode, "run": run_type}}
        )
        chain_success = True
        chain_error_comment = None
    except Exception as e:
        print("\n[ERROR] main_chain 실행 실패")
        print(f"- 이유: {e}")
        import traceback
        traceback.print_exc()
        chain_success = False
        chain_error_comment = str(e)
        result = {"chain1_out": "", "chain2_out": "", "chain3_out": "", "chain4_out": ""}

    t_chain_end = perf_counter()
    chain_elapsed = t_chain_end - t_chain_start

    # 3-1) chain4_out → 아두이노 전송 
    chain4_out = (result.get("chain4_out", "") or "").strip()
    print("\n----- Arduino 제어 시작 -----")
    try:
        RPI.connect_to_arduinos()

        connected = getattr(RPI, "arduino_connections", {})
        if not connected:
            print("[WARN] 연결된 아두이노가 없습니다. DRY-RUN 모드로 진행합니다.")
            print(f"[DRY-RUN] 16-digit code: {chain4_out}")
        else:
            time.sleep(0.3)
            if chain4_out:
                RPI.send_automated_command(chain4_out)
            time.sleep(2.0)

    except Exception as e:
        # 하드웨어 제어 중 예외가 나도 결과 저장은 계속 진행
        print(f"[WARN] 하드웨어 제어 중 예외 → DRY-RUN 전환: {e}")
        print(f"[DRY-RUN] 16-digit code: {chain4_out}")

    finally:
        try:
            RPI.close_all_connections()
        except Exception:
            pass
    print("----- Arduino 제어 종료 -----")

    # 4) 최종 출력 (로컬 파일 저장 유지)
    print("\n====================[ chain1_out ]====================")
    print(result.get("chain1_out", ""))
    print("\n====================[ chain2_out ]====================")
    print(result.get("chain2_out", ""))
    print("\n====================[ chain3_out ]====================")
    print(result.get("chain3_out", ""))
    print("\n====================[ chain4_out ]====================")
    print(chain4_out)

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

    # 5) (평가 모드일 때만) 지표 계산 → LangSmith Feedback 저장 (로컬 저장 X)
    if run_type == "eval" and os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        client = Client()
        project = os.getenv("LANGCHAIN_PROJECT", "tetris")

        # 최상위 성공 여부
        _push_feedback(client, project, common_tags, "tetris_success", 1.0 if chain_success else 0.0,
                       comment=chain_error_comment)

        # === Chain1: count accuracy ===
        try:
            gold1_path = AS_ROOT / "chain1" / f"{scenario}.json"
            if gold1_path.exists():
                gold1 = json.loads(gold1_path.read_text(encoding="utf-8"))
                gold_count = int(gold1["total_luggage_count"])
                r1 = _eval_chain1_count(result["chain1_out"], gold_count)
                _push_feedback(client, project, common_tags + ["chain1"], "chain1_success", r1.get("success", 0.0),
                               comment=r1.get("error"))
                if r1.get("success", 0.0) == 1.0:
                    _push_feedback(client, project, common_tags + ["chain1"], "chain1_count_accuracy", r1["count_accuracy"])
            else:
                _push_feedback(client, project, common_tags + ["chain1"], "chain1_success", 0.0,
                               comment="gold-missing")
        except Exception as e:
            _push_feedback(client, project, common_tags + ["chain1"], "chain1_success", 0.0,
                           comment=f"metric-error: {e}")

        # === Chain2: accuracy ===
        try:
            gold2_path = AS_ROOT / "chain2" / f"{scenario}.json"
            if gold2_path.exists():
                gold2 = json.loads(gold2_path.read_text(encoding="utf-8"))
                r2 = _eval_chain2_accuracy(result["chain2_out"], gold2)
                _push_feedback(client, project, common_tags + ["chain2"], "chain2_success", r2.get("success", 0.0),
                               comment=r2.get("error"))
                if r2.get("success", 0.0) == 1.0:
                    _push_feedback(client, project, common_tags + ["chain2"], "chain2_accuracy", r2["accuracy"])
            else:
                _push_feedback(client, project, common_tags + ["chain2"], "chain2_success", 0.0,
                               comment="gold-missing")
        except Exception as e:
            _push_feedback(client, project, common_tags + ["chain2"], "chain2_success", 0.0,
                           comment=f"metric-error: {e}")

        # === Chain3: precision/recall/f1 (시나리오+people 매칭) ===
        try:
            gold3 = _load_chain3_gold(scenario, int(people_count))
            if gold3 is not None:
                r3 = _eval_chain3_f1(result["chain3_out"], gold3)
                _push_feedback(client, project, common_tags + ["chain3"], "chain3_success", r3.get("success", 0.0),
                               comment=r3.get("error"))
                if r3.get("success", 0.0) == 1.0:
                    _push_feedback(client, project, common_tags + ["chain3"], "chain3_precision", r3["precision"])
                    _push_feedback(client, project, common_tags + ["chain3"], "chain3_recall", r3["recall"])
                    _push_feedback(client, project, common_tags + ["chain3"], "chain3_f1", r3["f1"])
            else:
                _push_feedback(client, project, common_tags + ["chain3"], "chain3_success", 0.0,
                               comment="gold-missing")
        except Exception as e:
            _push_feedback(client, project, common_tags + ["chain3"], "chain3_success", 0.0,
                           comment=f"metric-error: {e}")

        # === Chain4: 성공 여부(16자리 숫자)
        c4_ok = 1.0 if str(chain4_out).isdigit() and len(chain4_out) == 16 else 0.0
        _push_feedback(client, project, common_tags + ["chain4"], "chain4_success", c4_ok,
                       comment=None if c4_ok == 1.0 else "non-16-digit")

    return {
        "out_path": out_path,
        "chain_elapsed": chain_elapsed,
    }


def main():
    ap = argparse.ArgumentParser(description="AI TETRIS launcher")
    ap.add_argument("--mode", required=True, choices=["web", "scenario"])
    ap.add_argument("--port", type=int, default=5002)
    ap.add_argument("--no-browser", action="store_true")

    # 운영/평가, 추적, 모델/온도
    ap.add_argument("--run", choices=["operate","eval"], default="operate")
    ap.add_argument("--model", default=None)
    ap.add_argument("--temp", type=float, default=None)
    args = ap.parse_args()

    # === 정책 강제 ===
    if args.run == "eval":
        if args.mode != "scenario":
            print("❌ 평가모드는 'scenario' 모드에서만 가능합니다. (--mode scenario)")
            raise SystemExit(2)

    t_total_start = perf_counter()
    res = run_pipeline(
        mode=args.mode, port=args.port, open_browser=(not args.no_browser),
        run_type=args.run,
        model_override=args.model, temp_override=args.temp
    )
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
