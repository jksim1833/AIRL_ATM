# interface_test.py
# - UI만 확인하는 테스트 서버 (QR 페이지 + 인터페이스)
# - 변경 사항:
#   1) 탑승 인원 버튼에 "0명" 추가
#   2) JS에서 people 전송 조건을 seatSelection !== null 로 변경(0도 전송)

from __future__ import annotations
from flask import Flask, request, jsonify, make_response, render_template_string, send_from_directory, abort
from datetime import datetime
from pathlib import Path
from io import BytesIO
import socket, threading, time, argparse
import qrcode  # pip install qrcode[pil]
import mimetypes

app = Flask(__name__)

# 테스트용 자원 디렉토리(있으면 서빙, 없어도 무방)
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR  = (BASE_DIR / "user_input" / "web")  # 기존 구조 유지 시 자동 인식
WEB_DIR.mkdir(parents=True, exist_ok=True)

mimetypes.add_type("font/woff2", ".woff2")

# 간단한 메모리 상태
JOBS: dict[str, dict] = {}
LATEST_SCENARIO: str | None = None

APP_HTML = """<!doctype html>
<html lang="ko">
<head>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#ffffff">
<title>AI TETRIS</title>
<style>
  @font-face{
    font-family:"HyundaiHarmonyM";
    src:url("/web/HyundaiHarmonyM.woff2") format("woff2");
    font-weight:400; font-style:normal; font-display:swap;
  }
  @font-face{
    font-family:"HyundaiHarmonyL";
    src:url("/web/HyundaiHarmonyL.woff2") format("woff2");
    font-weight:300; font-style:normal; font-display:swap;
  }

  :root{
    --navy:#002c5f; --black:#000; --white:#fff; --dark:#374151; --light:#E5E7EB;
    --chipbg:#F3F4F6; --notice-bg: rgba(55,65,81,.9); --indent: 26px; --padX: 16px;
  }

  *{box-sizing:border-box}
  html,body{ margin:0; background:#fff; color:var(--black);
    font-family:"HyundaiHarmonyM","Noto Sans KR",system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    -webkit-font-smoothing:antialiased; }

  .topbar{ position:sticky; top:0; z-index:20; background:#fff;
    display:flex; align-items:center; justify-content:center;
    padding:10px var(--padX); border-bottom:1px solid var(--light); }
  .topbar h1{ margin:0; font-size:18px; color:var(--black); letter-spacing:.2px; }
  .topbar .close{ position:absolute; right:16px; top:50%; transform:translateY(-50%) scaleX(1.18);
    transform-origin:center; font-size:18px; color:var(--dark); text-decoration:none; font-weight:800; }

  .hero{ position:relative; width:100%; height:220px; overflow:hidden; background:#0f172a; margin-top:-4px; }
  .hero img{ width:100%; height:100%; object-fit:cover; object-position:50% 0%; display:block; }
  .hero-text{ position:absolute; left:16px; bottom:12px; color:var(--white);
    text-shadow:0 2px 8px rgba(0,0,0,.35); max-width:86%; }
  .hero-title{ margin:0 0 6px 0; line-height:1.12; font-family:"HyundaiHarmonyM"; }
  .hero-title .line1,.hero-title .line2{ display:block; font-size:26px; }
  .hero-sub{ margin:0; font-size:13px; line-height:1.4; font-family:"HyundaiHarmonyL"; }

  .wrap{ padding:8px var(--padX) 16px; }

  .section-head{ position:relative; padding-left:var(--indent); display:block; margin:8px 0 4px 0; }
  .section-head svg{ position:absolute; left:0; top:50%; transform:translateY(-50%); width:18px; height:18px; }
  .section-title{ margin:0; font-size:16px; color:var(--black); font-family:"HyundaiHarmonyM"; }

  .section-head + .section-desc{ margin-bottom:8px; }
  .section-desc{ margin:6px 0 6px var(--indent); color:var(--dark); line-height:1.22; font-family:"HyundaiHarmonyL";
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:clamp(11.4px, 3.1vw, 12.8px); }
  .section-desc.tight{ margin:2px 0 2px var(--indent); }
  .section-head + .section-desc.tight{ margin-bottom:0; }
  .section-desc.tight + .section-desc.tight{ margin-top:4px; }

  .chips{ display:grid; grid-template-columns:repeat(2, 1fr);
    column-gap:10px; row-gap:18px; margin-top:16px; }
  .chip{ display:flex; align-items:center; justify-content:center;
    height:44px; border-radius:22px; background:var(--chipbg);
    border:2px solid var(--navy); color:var(--navy); font-size:16px; font-weight:700; cursor:pointer; }
  .chip.selected{ background:var(--navy); color:#fff; }

  .divider{ height:3px; background:var(--light); margin:22px calc(-1 * var(--padX)); }

  .photo-box{ border:2px solid var(--light); border-radius:16px; padding:12px;
    display:flex; flex-direction:column; align-items:center; justify-content:center; gap:10px;
    min-height:180px; margin-top:14px; }
  #photo{ position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); border:0; }

  #btnPhotoIn{ display:inline-flex; align-items:center; justify-content:center; height:44px; padding:0 18px;
    border-radius:12px; border:2px solid var(--navy); background:var(--navy); color:#fff;
    font-size:15px; font-weight:800; cursor:pointer; width:72%; max-width:420px; text-align:center; }

  #btnPhotoOut{ display:none; margin-top:10px; margin-bottom:4px; width:100%; height:48px; padding:0 18px;
    border-radius:12px; border:2px solid var(--navy); background:var(--navy); color:#fff; font-size:16px; font-weight:900;
    cursor:pointer; align-items:center; justify-content:center; text-align:center; }

  #photo-preview{ display:none; max-width:100%; max-height:220px; border-radius:0; object-fit:contain;
    image-rendering:auto; image-orientation:from-image; }

  .cta{ margin:4px 0 24px; width:100%; height:48px; border-radius:14px; border:2px solid #cbd5e1;
    background:#e5e7eb; color:#475569; font-size:16px; font-weight:900; }
  .cta.active{ border-color:var(--navy); background:var(--navy); color:#fff; }

  .sr-only{ position:absolute; left:-10000px; }

  .notice{ position:fixed; left:50%; top:50%; transform:translate(-50%,-50%);
    padding:10px 18px; border-radius:10px; background:var(--notice-bg); color:#fff;
    box-shadow:0 10px 26px rgba(12,18,32,.18); font-size:13px; z-index:60; white-space:nowrap; text-align:center;
    min-width:min(80vw, 300px); max-width:90vw; font-family:"HyundaiHarmonyM"; opacity:0; pointer-events:none; }
  .notice.show{ animation:fadeInOut 2.8s ease-in-out forwards; }
  @keyframes fadeInOut{
    0%{opacity:0; transform:translate(-50%,-50%) scale(.98);}
    12%{opacity:1; transform:translate(-50%,-50%) scale(1);}
    80%{opacity:1;} 100%{opacity:0;}
  }
</style>
</head>
<body>

  <div class="topbar">
    <h1>AI TETRIS</h1>
    <a class="close" href="#" aria-label="닫기">X</a>
  </div>

  <section class="hero" aria-hidden="false">
    <img src="/web/AI_TETRIS.png" alt="AI TETRIS">
    <div class="hero-text">
      <h2 class="hero-title">
        <span class="line1">현대자동차</span>
        <span class="line2">AI TETRIS 서비스</span>
      </h2>
      <p class="hero-sub">단 한 장의 짐 사진으로 최적의 차량 시트 배치를<br>자동으로 완성하는 서비스입니다.</p>
    </div>
  </section>

  <main class="wrap">
    <div class="section-head">
      <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="8" r="4" fill="#9ca3af"/><rect x="5" y="13" width="14" height="8" rx="4" fill="#9ca3af"/></svg>
      <h3 class="section-title">탑승 인원</h3>
    </div>
    <p class="section-desc">1열을 제외한 차량 탑승 인원을 알려주세요.</p>

    <div class="chips" role="group" aria-label="탑승 인원 선택">
      <button class="chip" type="button" data-seats="0">0명</button> 
      <button class="chip" type="button" data-seats="1">1명</button>
      <button class="chip" type="button" data-seats="2">2명</button>
      <button class="chip" type="button" data-seats="3">3명</button>
      <button class="chip" type="button" data-seats="4">4명</button>
    </div>

    <div class="divider" aria-hidden="true"></div>

    <div class="section-head" style="margin-top:0">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 7l9-4 9 4-9 4-9-4zm0 4l9 4 9-4v7l-9 4-9-4v-7z" fill="#9ca3af"/></svg>
      <h3 class="section-title">짐</h3>
    </div>
    <p class="section-desc tight">대각선 방향에서 짐 이외의 배경은 최대한 보이지 않도록 촬영해주세요.</p>
    <p class="section-desc tight">개별 짐이 가려지는 부분이 없도록 주의하세요!</p>

    <div class="photo-box" aria-label="사진 촬영 영역">
      <img id="photo-preview" alt="업로드된 사진 미리보기">
      <input id="photo" type="file" accept="image/*" capture="environment">
      <label id="btnPhotoIn" for="photo">사진 촬영</label>
    </div>

    <label id="btnPhotoOut" for="photo">사진 촬영</label>

    <button id="submit" class="cta" disabled>최적 배치 시작</button>
    <p id="note" class="sr-only" aria-live="polite"></p>
  </main>

  <div id="notice" class="notice" role="status" aria-live="polite"></div>

<script>
/* 탑승 인원 선택 */
const chips = document.querySelectorAll('.chip');
let seatSelection = null;
chips.forEach(ch=>{
  ch.addEventListener('click', ()=>{
    chips.forEach(c=>c.classList.remove('selected'));
    ch.classList.add('selected');
    seatSelection = ch.dataset.seats; // 문자열 "0","1",...
  });
});

/* 파일/업로드/미리보기 */
const photo=document.querySelector('#photo'), submit=document.querySelector('#submit');
const notice=document.querySelector('#notice');
const preview=document.querySelector('#photo-preview');
const btnPhotoIn=document.querySelector('#btnPhotoIn');
const btnPhotoOut=document.querySelector('#btnPhotoOut');

let previewURL=null, currentScenario=null, pollTimer=null;

function showNotice(msg){
  notice.textContent = msg;
  notice.classList.remove('show'); void notice.offsetWidth; notice.classList.add('show');
}

/* 촬영 → 미리보기 → 자동 업로드 */
photo.addEventListener('change', async ()=>{
  if(!photo.files.length) return;

  if(previewURL) URL.revokeObjectURL(previewURL);
  previewURL = URL.createObjectURL(photo.files[0]);
  preview.src = previewURL;
  preview.style.display = 'block';

  try{
    const fd=new FormData();
    fd.append('photo', photo.files[0]);
    // ✅ 0도 전송되도록 null 체크(기존 truthy 체크 제거)
    if (seatSelection !== null) fd.append('people', seatSelection);

    const r=await fetch('/api/upload',{method:'POST',body:fd});
    const d=await r.json();
    if(!r.ok||!d.ok){ alert('업로드 실패'); return; }

    currentScenario = d.scenario;

    btnPhotoIn.style.display = 'none';
    btnPhotoOut.style.display = 'flex';

    showNotice('사진이 업로드 되었습니다!');
    submit.disabled = false;
    submit.classList.add('active');

  }catch(e){
    alert('네트워크 오류');
  }
});

/* CTA → 폴링 */
submit.addEventListener('click', ()=>{
  if(!currentScenario){ return; }
  showNotice('최적의 차량 배치 설계를 시작합니다!');
  let tries=0;
  clearInterval(pollTimer);
  pollTimer = setInterval(async ()=>{
    tries += 1;
    try{
      const rs = await fetch('/api/status?scenario='+encodeURIComponent(currentScenario));
      const dj = await rs.json();
      if(dj.ok && dj.status === 'done'){
        clearInterval(pollTimer);
        showNotice('테스트 완료: 상태 done');
      }else if(dj.ok && dj.status === 'error'){
        clearInterval(pollTimer);
        alert('분석 실패: ' + (dj.error_msg||''));
      }else if(tries>120){
        clearInterval(pollTimer);
        alert('분석 대기 시간이 초과되었습니다.');
      }
    }catch(e){
      clearInterval(pollTimer);
      alert('네트워크 오류');
    }
  }, 1500);
});
</script>
</body>
</html>"""

QR_HTML = """<!doctype html>
<html lang="ko">
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#000000">
<title>QR · AI TETRIS</title>
<style>
  @font-face{
    font-family:"HyundaiHarmonyM";
    src:url("/web/HyundaiHarmonyM.woff2") format("woff2");
    font-weight:400; font-style:normal; font-display:swap;
  }
  :root{ --qrw: min(70vmin, 520px); }
  html,body{
    height:100%; margin:0;
    background:radial-gradient(1200px 600px at 50% 40%, #0d1117 0%, #0b0e14 45%, #05070b 100%);
    color:#e6edf3; font-family:"HyundaiHarmonyM",system-ui,-apple-system,Segoe UI,Roboto,Noto Sans KR,sans-serif
  }
  .wrap{height:100%; display:grid; place-items:center; padding:18px}
  .box{display:flex; flex-direction:column; align-items:center; gap:14px}
  .label{
    background:#002c5f; color:#fff; padding:14px 18px; border-radius:18px;
    font-size:min(5.8vw,22px); font-weight:600; text-align:center; box-shadow:0 14px 34px rgba(0,0,0,.38);
    width:var(--qrw); font-family:"HyundaiHarmonyM";
  }
  .qr{
    width:var(--qrw); height:var(--qrw);
    background:#fff; border-radius:24px; padding:18px; box-shadow:0 22px 70px rgba(0,0,0,.55)
  }
</style>
</head>
<body>
  <div class="wrap">
    <div class="box">
      <div class="label">마이현대 앱에서 AI TETRIS를 경험해보세요!</div>
      <img class="qr" src="/qr.png?t={{ts}}" alt="접속 QR">
    </div>
  </div>
<script>
document.addEventListener('click', ()=>{
  if (document.fullscreenEnabled && !document.fullscreenElement) {
    document.documentElement.requestFullscreen()
      .catch(err => console.warn("전체화면 요청 실패:", err));
  }
});
</script>
</body>
</html>"""

def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def _server_url(port: int) -> str:
    return f"http://{_lan_ip()}:{port}"

@app.route("/", methods=["GET"])
def home():
    return APP_HTML

@app.route("/qr", methods=["GET"])
def qr_fullscreen():
    return render_template_string(QR_HTML, ts=time.time())

@app.route("/web/<path:filename>")
def web_assets(filename):
    p = WEB_DIR / filename
    if not p.exists():
        abort(404)
    return send_from_directory(WEB_DIR, filename)

@app.route("/qr.png")
def qr_png():
    # 현재 서버 루트 URL을 QR에 넣어 모바일 접속 확인
    port = int(request.host.split(":")[1]) if ":" in request.host else 5002
    url = _server_url(port)
    qr = qrcode.QRCode(box_size=10, border=2)
    qr.add_data(url); qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    buf = BytesIO(); img.save(buf, format="PNG")
    resp = make_response(buf.getvalue()); resp.headers["Content-Type"] = "image/png"; return resp

# ====== 테스트용 가짜 업로드/상태 엔드포인트 ======
@app.route("/api/upload", methods=["POST"])
def api_upload():
    global LATEST_SCENARIO
    scenario = datetime.now().strftime("items_%Y%m%d_%H%M%S")
    file = request.files.get("photo")
    if not file:
        return jsonify(ok=False, error="파일이 없습니다."), 400

    people_raw = (request.form.get("people") or "").strip()
    # 여기선 검증만, 실제 저장/처리는 하지 않음
    try:
        people_int = int(people_raw) if people_raw != "" else None  # "0"도 int(0)로 처리
    except Exception:
        people_int = None

    JOBS[scenario] = {
        "status": "uploaded",
        "people": people_int,   # 0도 올바르게 들어옴
        "error_msg": None,
    }
    LATEST_SCENARIO = scenario
    return jsonify(ok=True, stage="uploaded", scenario=scenario)

@app.route("/api/status", methods=["GET"])
def api_status():
    scenario = (request.args.get("scenario") or "").strip()
    if not scenario or scenario not in JOBS:
        return jsonify(ok=False, error="unknown scenario"), 404

    job = JOBS[scenario]
    # 업로드 후 첫 조회에 done으로 바꿔주는 간단한 모의 동작
    if job["status"] == "uploaded":
        job["status"] = "done"

    return jsonify(ok=True, status=job["status"], error_msg=job.get("error_msg"))

def main():
    parser = argparse.ArgumentParser(description="AI TETRIS UI Interface Test")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    # 안내 출력
    print(f"\n[AI TETRIS · UI TEST] 휴대폰 접속:  http://{_lan_ip()}:{args.port}\n(같은 Wi-Fi 필요)")
    if not args.no_browser:
        try:
            import webbrowser
            webbrowser.open(f"http://127.0.0.1:{args.port}/qr")
        except Exception:
            pass

    # 플라스크 실행
    app.run(host="0.0.0.0", port=args.port, debug=False)

if __name__ == "__main__":
    main()
