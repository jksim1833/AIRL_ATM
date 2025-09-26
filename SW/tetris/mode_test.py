# mode_test.py  (최종본: create_run -> HTTP 폴백 -> run discovery 강화)
from __future__ import annotations
import os, json, re, traceback, requests, time
from time import perf_counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
import sys

# 초기 로드 & 디버그
HERE = Path(__file__).resolve().parent
print(f"[DEBUG] script_dir={HERE} cwd={Path.cwd()} .env_exists={(HERE/'.env').exists()}")
load_dotenv(override=True)
print("[DEBUG] Loaded .env via python-dotenv")
print(f"[DEBUG] LANGSMITH_API_KEY (masked): {'SET' if os.getenv('LANGSMITH_API_KEY') else 'NOT SET'}")
print(f"[DEBUG] LANGCHAIN_PROJECT: {os.getenv('LANGCHAIN_PROJECT')}")
sys.stdout.flush()

# 경로 보정
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
UI_DIR = (HERE / "user_input").resolve()
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

# 모듈 임포트
import tetris
from user_input import file_to_data_url, input_scenario_image, input_scenario_people

try:
    from langsmith import Client
except Exception:
    Client = None
    print("[WARN] langsmith import failed; SDK features disabled. HTTP fallback only.")
    sys.stdout.flush()

AS_ROOT = HERE / "answer_sheet"
IMG_DIR = (HERE / "user_input" / "luggage_image").resolve()

# 프롬프트 파일들 (대시보드 업로드용 텍스트)
MC_DIR = HERE / "main_chain"
CHAIN1_PROMPT_TXT = MC_DIR / "chain1_prompt" / "chain1_prompt.txt"
CHAIN2_PROMPT_DIR = MC_DIR / "chain2_prompt"
CHAIN2_PROMPT_TXT = CHAIN2_PROMPT_DIR / "chain2_prompt.txt"
CHAIN3_DIR = MC_DIR / "chain3_prompt"
C3_SYSTEM_TXT = CHAIN3_DIR / "chain3_system.txt"
C3_QUERY_TXT = CHAIN3_DIR / "chain3_query.txt"
C3_ROLE_TXT = CHAIN3_DIR / "chain3_prompt_role.txt"
C3_ENV_TXT = CHAIN3_DIR / "chain3_prompt_environment.txt"
C3_FUNC_TXT = CHAIN3_DIR / "chain3_prompt_function.txt"
C3_OUTFMT_TXT = CHAIN3_DIR / "chain3_prompt_output_format.txt"
C3_EXAMPLE_TXT = CHAIN3_DIR / "chain3_prompt_example.txt"

SCENARIOS: List[str] = [
    "0", "1-1", "1-2", "1-3", "1-4", "1-5", "2", "3", "4", "5",
    "6-1", "6-2", "7", "8-1", "8-2", "9", "10-1", "10-2",
    "11-1", "11-2", "12", "13", "14", "15", "16", "17-1", "17-2", "18", "19"
]

# --- 유틸 (원본 유지) ---
def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return f"[missing or unreadable: {p.name}]"

def _norm_json(s: str) -> dict:
    s = (s or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S | re.I)
    if m: s = m.group(1).strip()
    if not (s.startswith("{") and s.endswith("}")):
        first = s.find("{"); last = s.rfind("}")
        if first != -1 and last != -1 and first < last:
            s = s[first:last+1]
    return json.loads(s)

def _canon(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _load_json_if_exists(p: Path) -> Optional[dict]:
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def _chain2_gold_candidates(scenario: str, people: int) -> List[Tuple[str, dict]]:
    out: List[Tuple[str, dict]] = []
    base1 = AS_ROOT / "chain2" / f"people_{people}"
    base2 = AS_ROOT / "chain2"
    for base in (base1, base2):
        for name in (f"{scenario}.json", f"{scenario}_1.json", f"{scenario}_2.json"):
            obj = _load_json_if_exists(base / name)
            label = name.replace(".json","")
            if obj is not None and label not in [l for l,_ in out]:
                out.append((label, obj))
    return out

def _chain3_gold_candidates(scenario: str, people: int) -> List[Tuple[str, dict]]:
    out: List[Tuple[str, dict]] = []
    base1 = AS_ROOT / "chain3" / f"people_{people}"
    base2 = AS_ROOT / "chain3" / scenario / f"people_{people}"
    for base in (base1, base2):
        for name in (f"{scenario}.json", f"{scenario}_1.json", f"{scenario}_2.json"):
            obj = _load_json_if_exists(base / name)
            label = name.replace(".json","")
            if obj is not None and label not in [l for l,_ in out]:
                out.append((label, obj))
    return out

# 평가 지표 함수들 (원본 유지)
def _eval_chain1_count(chain1_out_text: str, gold_count: int) -> dict:
    try:
        d = json.loads(chain1_out_text)
        pred = int(d.get("total_luggage_count", -999))
        acc = 1.0 if pred == gold_count else 0.0
        return {"success": 1.0, "count_pred": pred, "count_accuracy": acc}
    except Exception as e:
        return {"success": 0.0, "error": f"parse-error: {e}"}

def _normalize_seq_tokens_from_obj(obj: dict) -> set[str]:
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

# 이미지 로더
def _image_data_url_for(scenario: str) -> Optional[str]:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = IMG_DIR / f"{scenario}{ext}"
        if p.exists():
            return file_to_data_url(p)
    return None

# 입력 함수들 (원본 유지)
def input_repeat_count(prompt: str = "반복 횟수를 입력하세요: ") -> int:
    while True:
        s = input(prompt).strip()
        try:
            n = int(s)
            if n <= 0:
                print("❌ 1 이상의 정수를 입력해주세요.")
                continue
            while True:
                c = input(f'반복 횟수는 "{n}회"가 맞나요? (1) 네 (2) 아니요 : ').strip()
                if c == "1":
                    return n
                elif c == "2":
                    break
                else:
                    print("❌ 1 또는 2로 입력해주세요.")
        except ValueError:
            print("❌ 숫자만 입력해주세요.")

def input_mode_choice() -> int:
    print("\n=== [평가 모드] 실행 ===")
    print("\n===================== 평가 모드 실행 방식 선택 =====================")
    print(" (1) 단일 시나리오 N회 반복")
    print(" (2) 전체 시나리오 N회 반복")
    while True:
        s = input("번호를 선택하세요: ").strip()
        if s in ("1","2"):
            return int(s)
        print("❌ 1 또는 2를 입력해주세요.")

def input_people_uniform_for_all() -> int:
    while True:
        s = input("전체 시나리오에 적용할 공통 people_count를 입력하세요(0~4): ").strip()
        try:
            k = int(s)
            if 0 <= k <= 4:
                print(f"✅ 모든 시나리오에 people_count={k} 적용합니다.")
                return k
        except ValueError:
            pass
        print("❌ 0~4의 정수를 입력해주세요.")

def input_people_for_trials(n: int) -> List[int]:
    arr: List[int] = []
    print(f"\ntrial마다 다른 people_count를 입력합니다. (0~4)")
    for i in range(1, n + 1):
        while True:
            s = input(f" - Trial {i}의 people_count: ").strip()
            try:
                k = int(s)
                if 0 <= k <= 4:
                    arr.append(k)
                    break
            except ValueError:
                pass
            print("  ❌ 0~4의 정수를 입력해주세요.")
    print("\n입력 요약:")
    for i, v in enumerate(arr, 1):
        print(f"  Trial {i}: people_count={v}")
    while True:
        c = input("이대로 진행할까요? (1) 네 (2) 다시 입력 : ").strip()
        if c == "1":
            return arr
        elif c == "2":
            return input_people_for_trials(n)

# LangSmith helpers (강화된 create + discovery)
def _build_trace_inputs_for_dashboard(scenario: str, people_count: int) -> dict:
    return {
        "scenario": scenario,
        "people_count": people_count,
        "note": "image_data_url omitted from dashboard upload",
        "chain1_prompt_system": _read_text(CHAIN1_PROMPT_TXT),
        "chain2_prompt_system": _read_text(CHAIN2_PROMPT_TXT),
        "chain3_system":  _read_text(C3_SYSTEM_TXT),
        "chain3_role":    _read_text(C3_ROLE_TXT),
        "chain3_env":     _read_text(C3_ENV_TXT),
        "chain3_func":    _read_text(C3_FUNC_TXT),
        "chain3_outfmt":  _read_text(C3_OUTFMT_TXT),
        "chain3_example": _read_text(C3_EXAMPLE_TXT),
        "chain3_query":   _read_text(C3_QUERY_TXT),
    }

def _discover_run_via_http(api_url: str, api_key: str, project: str, name: str, timeout_s: int = 30, poll_interval: float = 2.0):
    """
    HTTP로 /runs 리스트를 반복 조회하여 name과 유사한 run을 찾고 id 반환.
    (서버가 비동기로 run을 생성할 때 유용)
    """
    url = api_url.rstrip("/") + "/runs"
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    deadline = time.time() + timeout_s
    print(f"[INFO] Attempting to discover run via list_runs for name={name} (timeout {timeout_s}s)...")
    while time.time() < deadline:
        try:
            # server may accept query params; try simple list first
            resp = requests.get(url, headers=headers, params={"project_name": project, "limit": 200}, timeout=10)
            if resp.status_code in (200, 201, 202):
                try:
                    j = resp.json()
                except Exception:
                    j = None
                # j might be dict with 'data' or list — try to inspect
                candidates = []
                if isinstance(j, dict):
                    # common shapes: {"runs": [...]} or {"data": [...]} or list in some key
                    for key in ("runs", "data", "items"):
                        if key in j and isinstance(j[key], list):
                            candidates = j[key]
                            break
                    if not candidates:
                        # maybe direct list fields
                        for v in j.values():
                            if isinstance(v, list):
                                candidates = v
                                break
                elif isinstance(j, list):
                    candidates = j

                if candidates:
                    # try to match by name (exact or contains) and project
                    for item in candidates:
                        if not isinstance(item, dict):
                            continue
                        rn = item.get("name") or item.get("display_name") or item.get("title")
                        rproj = item.get("project_name") or item.get("project")
                        if rn and name in rn and (not project or project == rproj):
                            rid = item.get("id") or item.get("run_id")
                            if rid:
                                print("[INFO] Discovered run by listing:", rid)
                                return rid, item
            else:
                print("[DEBUG] list_runs returned status", resp.status_code, resp.text[:300])
        except Exception as e:
            print("[DEBUG] list_runs exception:", e)
        time.sleep(poll_interval)
    print("[WARN] discover_run_via_http: no matching run found within timeout.")
    return None, None


def _safe_create_run(client, project: str, name: str, inputs: dict, tags: list[str]):
    """
    Robust create_run:
      - try SDK.create_run() first (dump everything)
      - if no id, try HTTP fallback using x-api-key
      - if HTTP returned success but no id, attempt to discover run by listing (poll for up to 30s)
      - always return an object/dict or None
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    api_url = (os.getenv("LANGSMITH_API_URL") or "https://api.smith.langchain.com").rstrip("/")

    # 1) SDK attempt
    try:
        run = client.create_run(
            run_type="chain",
            project_name=project,
            name=name,
            inputs=inputs,
            tags=tags,
            extra={"runtime": {"name": "tetris-eval"}},
            start_time=datetime.now(timezone.utc),
        )
        print("[DEBUG] SDK create_run returned (type):", type(run))
        try:
            print("[DEBUG] repr(run):", repr(run))
        except Exception:
            pass

        if isinstance(run, dict):
            try:
                print("[DEBUG] SDK run (json):\n", json.dumps(run, ensure_ascii=False, indent=2))
            except Exception:
                print("[DEBUG] SDK run (raw dict):", run)
        else:
            # try introspect attributes
            try:
                attrs = {}
                for a in dir(run):
                    if a.startswith("_"):
                        continue
                    try:
                        v = getattr(run, a)
                        if callable(v):
                            continue
                        attrs[a] = v
                    except Exception:
                        attrs[a] = "<error>"
                print("[DEBUG] SDK run attributes dump:", attrs)
            except Exception:
                pass

        rid = None
        if isinstance(run, dict):
            rid = run.get("id") or run.get("run_id")
        else:
            rid = getattr(run, "id", None) or getattr(run, "run_id", None)

        if rid:
            print("[INFO] SDK returned run_id:", rid)
            return run

        print("[WARN] SDK returned no id; will try HTTP fallback.")
    except Exception as e:
        print("[WARN] client.create_run raised exception:", e)
        traceback.print_exc()

    # 2) HTTP fallback using x-api-key
    try:
        url = api_url + "/runs"
        headers = {"x-api-key": api_key, "Content-Type": "application/json"}
        payload = {
            "run_type": "chain",
            "project_name": project,
            "name": name,
            "inputs": inputs,
            "tags": tags,
            "extra": {"runtime": {"name": "tetris-eval"}},
            "start_time": datetime.now(timezone.utc).isoformat(),
        }
        print("[INFO] Attempting HTTP fallback create_run (x-api-key) ->", url)
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        print("[DEBUG] HTTP create_run status:", resp.status_code)
        body_text = ""
        try:
            body_text = resp.text
            print("[DEBUG] HTTP create_run body (raw):", body_text)
            try:
                body_json = resp.json()
                print("[DEBUG] HTTP create_run body (json):\n", json.dumps(body_json, ensure_ascii=False, indent=2))
            except Exception:
                body_json = None
        except Exception as e:
            print("[WARN] could not read response body:", e)
            body_json = None

        if resp.status_code in (200, 201, 202):
            # if id present, return
            if isinstance(body_json, dict):
                rid = body_json.get("id") or body_json.get("run_id")
                if rid:
                    return {"id": rid, "raw": body_json}
            # no id — attempt to discover by listing
            discovered_id, discovered_obj = _discover_run_via_http(api_url, api_key, project, name, timeout_s=30)
            if discovered_id:
                return {"id": discovered_id, "raw": discovered_obj or body_json}
            # still nothing -> return the raw HTTP response as evidence
            return {"created_raw": body_json or body_text, "http_status": resp.status_code}
        else:
            print("[WARN] HTTP fallback create_run non-success:", resp.status_code)
            return {"http_status": resp.status_code, "body": body_text}
    except Exception as e:
        print("[ERROR] HTTP fallback create_run exception:", e)
        traceback.print_exc()
        return None


def _safe_update_run(client, run_id: Optional[str], outputs: dict, error: Optional[str] = None):
    """
    Try to update run via SDK; if run_id is None or SDK/update fails, try HTTP fallback.
    If run_id is None, write local backup JSON and return False.
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    api_url = (os.getenv("LANGSMITH_API_URL") or "https://api.smith.langchain.com").rstrip("/")

    if not run_id:
        print("[WARN] run_id is None — cannot update remote run. Saving local backup.")
        OUT_DIR = (Path(__file__).resolve().parent / "tetris_out" / "failed_langsmith_runs")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_path = OUT_DIR / f"run_missing_{ts}.json"
        try:
            out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
            print("[INFO] Saved local backup:", out_path)
        except Exception as e:
            print("[ERROR] Failed to write local backup:", e)
        return False

    # try SDK update_run
    try:
        try:
            resp = client.update_run(run_id, outputs=outputs, error=error, end_time=datetime.now(timezone.utc))
            print("[DEBUG] update_run (positional) response:", repr(resp))
            try:
                client.flush()
            except Exception:
                pass
            return True
        except TypeError:
            resp = client.update_run(run_id=run_id, outputs=outputs, error=error, end_time=datetime.now(timezone.utc))
            print("[DEBUG] update_run (keyword) response:", repr(resp))
            try:
                client.flush()
            except Exception:
                pass
            return True
    except Exception as e:
        print("[WARN] SDK update_run failed:", e)
        traceback.print_exc()

    # HTTP fallback to patch runs/{run_id}
    try:
        url = f"{api_url}/runs/{run_id}"
        headers = {"x-api-key": api_key, "Content-Type": "application/json"}
        payload = {"outputs": outputs, "error": error, "end_time": datetime.now(timezone.utc).isoformat()}
        print("[INFO] Attempting HTTP fallback update_run ->", url)
        resp = requests.patch(url, headers=headers, json=payload, timeout=20)
        print("[DEBUG] HTTP update_run status:", resp.status_code, "body:", resp.text[:300])
        if resp.status_code in (200, 201, 202):
            return True
        else:
            print("[WARN] HTTP update_run non-success:", resp.status_code)
            return False
    except Exception as e:
        print("[ERROR] HTTP fallback update_run exception:", e)
        traceback.print_exc()
        return False

# ---------------- run_once_eval (기존 로직 유지하되 _safe_create/_safe_update 사용) ----------------
def run_once_eval(
    scenario: str,
    people_count: int,
    image_data_url: str,
    *,
    trial_index: int,
    project_name: str = "tetris",
    extra_tags: Optional[List[str]] = None,
) -> dict:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    tags = [
        "tetris",
        f"scenario:{scenario}",
        f"people:{people_count}",
        "run:eval",
        f"trial:{trial_index}",
    ] + (extra_tags or [])
    client = Client() if Client else None
    project = os.getenv("LANGCHAIN_PROJECT", project_name)
    trace_inputs = _build_trace_inputs_for_dashboard(scenario, people_count)

    # create run (safe)
    name = f"{scenario} / people:{people_count} / trial:{trial_index}"
    print("[INFO] Creating LangSmith run:", name)
    run = _safe_create_run(client, project, name, trace_inputs, tags)
    print("[DEBUG] create_run raw result:", repr(run))
    run_id = None
    if run:
        if isinstance(run, dict):
            run_id = run.get("id") or run.get("run_id") or None
        else:
            run_id = getattr(run, "id", None)
    print("[DEBUG] resolved run_id:", repr(run_id))
    sys.stdout.flush()

    # run pipeline
    t0 = perf_counter()
    try:
        result = tetris.run_pipeline(
            mode="scenario",
            override={"people_count": people_count, "image_data_url": image_data_url, "scenario": scenario},
            langsmith_tags=tags,
            langsmith_metadata={"scenario": scenario, "people": people_count, "mode": "scenario", "run": "eval"},
            eval_dry_run_hw=True,
        )
        chain_success = True
        chain_err = None
    except Exception as e:
        print("[ERROR] tetris.run_pipeline 실패:", e)
        traceback.print_exc()
        result = {"chain1_out": "", "chain2_out": "", "chain3_out": "", "chain4_out": ""}
        chain_success = False
        chain_err = str(e)
    t1 = perf_counter()
    chain_elapsed = t1 - t0

    outputs = {
        "chain1_out": result.get("chain1_out", ""),
        "chain2_out": result.get("chain2_out", ""),
        "chain3_out": result.get("chain3_out", ""),
        "chain4_out": result.get("chain4_out", ""),
        "chain_elapsed_sec": chain_elapsed,
        "error": chain_err,
    }

    # metrics (원본과 동일)
    try:
        gold1_path = AS_ROOT / "chain1" / f"{scenario}.json"
        if gold1_path.exists():
            gold1 = json.loads(gold1_path.read_text(encoding="utf-8"))
            r1 = _eval_chain1_count(result.get("chain1_out", ""), int(gold1["total_luggage_count"]))
            outputs["metrics_chain1"] = r1
    except Exception:
        pass

    try:
        c2_cands = _chain2_gold_candidates(scenario, people_count)
        if c2_cands:
            pred_canon = _canon(_norm_json(result.get("chain2_out", "")))
            acc = 0.0
            matched_label = None
            for idx, (_, g) in enumerate(c2_cands):
                if _canon(g) == pred_canon:
                    acc = 1.0
                    matched_label = c2_cands[idx][0]
                    break
            outputs["metrics_chain2"] = {"accuracy": acc, "matched_label": matched_label}
    except Exception:
        pass

    try:
        chain3_text = result.get("chain3_out", "")
        c3_cands = _chain3_gold_candidates(scenario, people_count)
        if c3_cands:
            match_idx = None
            try:
                c2_cands = _chain2_gold_candidates(scenario, people_count)
                if c2_cands:
                    pred_canon = _canon(_norm_json(result.get("chain2_out", "")))
                    for idx, (_, g) in enumerate(c2_cands):
                        if _canon(g) == pred_canon:
                            match_idx = idx
                            break
            except Exception:
                match_idx = None

            if match_idx is not None and match_idx < len(c3_cands):
                label, gold3 = c3_cands[match_idx]
                r3 = _eval_chain3_f1(chain3_text, gold3)
                outputs["metrics_chain3"] = {
                    "precision": r3.get("precision"),
                    "recall": r3.get("recall"),
                    "f1": r3.get("f1"),
                    "matched_label": label,
                }
            else:
                best = None; best_label = None
                for label, gold3 in c3_cands:
                    r = _eval_chain3_f1(chain3_text, gold3)
                    if r.get("success",0.0) == 1.0:
                        if not best or r["f1"] > best["f1"]:
                            best, best_label = r, label
                if best:
                    outputs["metrics_chain3"] = {
                        "precision": best["precision"], "recall": best["recall"], "f1": best["f1"],
                        "matched_label": best_label, "mode": "max-of-two"
                    }
    except Exception:
        pass

    # update run (safe)
    try:
        ok = _safe_update_run(client, run_id, outputs, error=None if chain_success else (chain_err or "unknown-error"))
        if not ok:
            print("[WARN] LangSmith update skipped/failed (local backup saved).")
    except Exception as e:
        print("[ERROR] update_run flow exception:", e)
        traceback.print_exc()

    return {"chain_elapsed": chain_elapsed}

# ---------------- main 인터랙티브 (원본 로직 유지) ----------------
def main():
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    try:
        mode = input_mode_choice()
        project = os.getenv("LANGCHAIN_PROJECT", "tetris")
        if mode == 1:
            scenario = input_scenario_image()
            data_url = _image_data_url_for(scenario)
            if not data_url:
                print(f"❌ 이미지 파일을 찾을 수 없습니다: {IMG_DIR / (scenario + '.jpg')}")
                return
            repeat_n = input_repeat_count("해당 시나리오를 몇 번 반복 실행할까요? : ")
            if repeat_n == 1:
                people = input_scenario_people()
                t0 = perf_counter()
                print(f"\n--- Trial 1/1 · scenario={scenario} · people={people} ---")
                res = run_once_eval(scenario=scenario, people_count=people, image_data_url=data_url, trial_index=1, project_name=project)
                print(f"🕒 chain_run_time: {res['chain_elapsed']:.3f}s")
                t1 = perf_counter()
            else:
                yn = input("사람 수를 모든 실행에 동일하게 적용할까요? (Y/N): ").strip().lower()
                if yn in ("", "y", "yes"):
                    people = input_scenario_people()
                    t0 = perf_counter()
                    for trial in range(1, repeat_n + 1):
                        print(f"\n--- Trial {trial}/{repeat_n} · scenario={scenario} · people={people} ---")
                        res = run_once_eval(scenario=scenario, people_count=people, image_data_url=data_url, trial_index=trial, project_name=project)
                        print(f"🕒 chain_run_time: {res['chain_elapsed']:.3f}s")
                    t1 = perf_counter()
                else:
                    people_list = input_people_for_trials(repeat_n)
                    t0 = perf_counter()
                    for trial in range(1, repeat_n + 1):
                        people = people_list[trial - 1]
                        print(f"\n--- Trial {trial}/{repeat_n} · scenario={scenario} · people={people} ---")
                        res = run_once_eval(scenario=scenario, people_count=people, image_data_url=data_url, trial_index=trial, project_name=project)
                        print(f"🕒 chain_run_time: {res['chain_elapsed']:.3f}s")
                    t1 = perf_counter()
            print("\n====================[ 평가 실행 완료 ]====================")
            print(f"총 실행: {repeat_n}회  ·  총 소요: {t1 - t0:.3f}s")
        else:
            repeat_rounds = input_repeat_count("전체 시나리오 라운드를 몇 번 반복할까요? : ")
            people_common = input_people_uniform_for_all()
            img_map: Dict[str, Optional[str]] = {}
            missing: List[str] = []
            for sc in SCENARIOS:
                url = _image_data_url_for(sc)
                img_map[sc] = url
                if not url:
                    missing.append(sc)
            if missing:
                print("\n⚠️ 다음 시나리오는 이미지가 없어 건너뜁니다:")
                print("   " + ", ".join(missing))
            total_runs = 0
            t_all0 = perf_counter()
            for r in range(1, repeat_rounds + 1):
                print(f"\n========== Round {r}/{repeat_rounds} (전체 시나리오 1회씩) ==========")
                for sc in SCENARIOS:
                    if sc in missing:
                        print(f"- {sc}: 이미지 없음 → SKIP")
                        continue
                    total_runs += 1
                    tags_extra = [f"round:{r}"]
                    print(f"\n--- {sc} · round={r} · people={people_common} ---")
                    res = run_once_eval(scenario=sc, people_count=people_common, image_data_url=img_map[sc] or "", trial_index=r, project_name=os.getenv("LANGCHAIN_PROJECT","tetris"), extra_tags=tags_extra)
                    print(f"🕒 chain_run_time: {res['chain_elapsed']:.3f}s")
            t_all1 = perf_counter()
            print("\n====================[ 전체 라운드 실행 완료 ]====================")
            print(f"총 실행: {total_runs}회  ·  총 소요: {t_all1 - t_all0:.3f}s")
    except Exception as e:
        print("[FATAL] main() exception:", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
