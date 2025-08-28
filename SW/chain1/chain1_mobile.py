# -*- coding: utf-8 -*-
"""
chain1_mobile.py (Hyundai-style, 2단계 알림, 로고 적용, '원본 그대로' 저장)
- 휴대폰: 사진 촬영/선택 → '이미지 업로드' → 업로드 즉시 토스트(이미지가 업로드 되었습니다!) → 분석 완료 토스트(이미지 분석이 완료되었습니다!)
- 저장 규칙:
    * chain1_image/: <scenario>.<원본확장자>   (이미지 1개만, 리사이즈/재인코딩/회전보정 없음 = 원본 바이트 그대로)
    * chain1_out/  : <scenario>.txt           (텍스트만)
- 상단 현대 로고(/brand.png), 현대 톤 컬러/타이포, 앱형 레이아웃(앱바/히어로/카드/탭)
- /qr (풀스크린 QR): '마이현대 앱으로 접속하기' + QR만

필요:
    pip install flask pillow pillow-heif qrcode[pil]
"""

from pathlib import Path
from datetime import datetime
from io import BytesIO
from flask import Flask, request, jsonify, make_response, render_template_string
from PIL import Image, ImageDraw, ImageFont
import socket, threading, mimetypes

# HEIC/HEIF 지원
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

# QR
try:
    import qrcode
    QR_AVAILABLE = True
except Exception:
    QR_AVAILABLE = False

# 분석기
from chain1 import LuggageAnalyzer

app = Flask(__name__)

# 경로
DESKTOP = Path.home() / "Desktop"
BASE    = DESKTOP / "AIRL_ATM" / "SW" / "chain1"
IMG_DIR = BASE / "chain1_image"
OUT_DIR = BASE / "chain1_out"
BRAND_DIR = BASE / "brand"
IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
BRAND_DIR.mkdir(parents=True, exist_ok=True)

# 설정
PORT = 5002
AUTO_OPEN_BROWSER = True
AUTO_OPEN_BROWSER_PATH = "/qr"  # PC에서 서버 시작 시 /qr 자동 오픈

# 업로드 → 백그라운드 분석 상태 관리
JOBS = {}  # scenario -> {"status": "uploaded"/"processing"/"done"/"error", "out_txt": str|None, "error_msg": str|None}

# ====================== 프론트 (Hyundai-style) ======================
APP_HTML = """
<!doctype html>
<html lang="ko">
<head>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#ffffff">
<title>myHyundai · AI TETRIS</title>
<style>
  :root{
    --brand:#002C5F; --brand2:#0B2E6B; --ink:#0b1220; --muted:#6b768a;
    --bg:#f6f7fb; --card:#ffffff; --line:#e9edf3; --tab:#95a0b3;
  }
  *{box-sizing:border-box}
  html,body{margin:0;background:var(--bg);color:var(--ink);
            font-family:system-ui,-apple-system,Segoe UI,Roboto,Apple SD Gothic Neo,Noto Sans KR,sans-serif}

  /* AppBar */
  .appbar{position:sticky;top:0;z-index:20;background:#fff;
          padding:12px 16px;border-bottom:1px solid var(--line);
          display:flex;align-items:center;justify-content:space-between}
  .brand{display:flex;align-items:center;gap:10px}
  .brand img{height:22px}
  .ab-icons{display:flex;gap:16px;color:#2f3a52;font-size:18px}

  /* Hero */
  .hero{position:relative; margin:0; height:220px; overflow:hidden; background:linear-gradient(180deg,#e9eef7,#ffffff)}
  .hero .bg{
    position:absolute; inset:0;
    background: radial-gradient(60% 80% at 70% 50%, #eef3fb 0%, #ffffff 60%, #ffffff 100%);
  }
  .hero .copy{
    position:absolute; left:16px; bottom:28px; color:#fff;
    text-shadow:0 2px 8px rgba(0,0,0,.35);
  }
  .hero .title{font-size:28px; font-weight:900; letter-spacing:.2px}
  .hero .sub{margin-top:6px; font-size:14px; opacity:.9}
  .hero .car{position:absolute; right:-20px; bottom:-10px; width:65%; max-width:420px; opacity:.95; filter:drop-shadow(0 18px 28px rgba(0,0,0,.22))}
  .car svg{width:100%; height:auto}

  .wrap{padding:16px}

  /* Upload Card */
  .card{background:var(--card); border:1px solid var(--line); border-radius:18px; padding:18px;
        box-shadow:0 10px 26px rgba(12,18,32,.06); max-width:560px; margin:0 auto 14px auto}
  .h1{font-size:18px; font-weight:800; margin:4px 2px 4px}
  .p {font-size:14px; color:var(--muted); margin:0 2px 14px}

  .row{display:flex; gap:10px; align-items:center}
  .row input[type=text]{
    flex:1; padding:12px 12px; border:1px solid var(--line); border-radius:12px; background:#fff; outline:none;
  }

  .filebox{margin-top:12px;border:1.5px dashed #cfd8e3;border-radius:14px;padding:18px;text-align:center;background:#fff}
  .filebox input{display:none}
  .filebtn{display:inline-block;padding:12px 16px;border-radius:12px;background:linear-gradient(135deg,var(--brand),var(--brand2));color:#fff;font-weight:800;cursor:pointer}

  .primary{width:100%;margin-top:12px;padding:13px 16px;border:none;border-radius:14px;
           background:linear-gradient(135deg,var(--brand),var(--brand2));color:#fff;font-weight:800;letter-spacing:.2px;cursor:pointer}
  .primary:disabled{opacity:.5}

  /* Bottom Tabs (데모) */
  .tabs{position:sticky; bottom:0; background:#fff; border-top:1px solid var(--line); display:flex}
  .tab{flex:1;text-align:center;padding:10px 4px;color:var(--tab);font-size:11px}
  .tab .ico{display:block; font-size:18px; margin-bottom:2px}
  .tab.active{color:#1e293b; font-weight:700}

  /* Toast */
  .toast{position:fixed;left:50%;bottom:22px;transform:translateX(-50%);
         background:rgba(0,0,0,.86);color:#fff;padding:10px 12px;border-radius:12px;font-size:13px;display:none}

  .sr-only{position:absolute;left:-10000px}
</style>
</head>
<body>

  <!-- AppBar -->
  <header class="appbar">
    <div class="brand"><img src="/brand.png" alt="Hyundai"></div>
    <div class="ab-icons">☰ 🔔</div>
  </header>

  <!-- Hero -->
  <section class="hero" aria-hidden="true">
    <div class="bg"></div>
    <div class="copy">
      <div class="title">The all-new NEXO</div>
      <div class="sub">당신만이 할 수 있는 일</div>
    </div>
    <div class="car">
      <svg viewBox="0 0 800 300" role="img" aria-label="vehicle">
        <defs><linearGradient id="g" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#F6F9FD"/><stop offset="100%" stop-color="#E9EEF7"/></linearGradient></defs>
        <rect x="0" y="200" width="800" height="50" fill="#dfe7f3"/>
        <path d="M60 210 C120 110, 220 80, 320 80 L520 80 C650 80, 700 140, 740 210 Z" fill="url(#g)" stroke="#c7d2e5"/>
        <circle cx="260" cy="220" r="36" fill="#0f172a"/><circle cx="260" cy="220" r="18" fill="#334155"/>
        <circle cx="590" cy="220" r="36" fill="#0f172a"/><circle cx="590" cy="220" r="18" fill="#334155"/>
      </svg>
    </div>
  </section>

  <!-- Upload Card -->
  <main class="wrap">
    <section class="card" aria-labelledby="upload-title">
      <h2 id="upload-title" class="h1">AI TETRIS</h2>
      <p class="p">사진 한 장으로 최적의 차량 배치가 완성돼요!</p>

      <div class="row">
        <input id="scenario" type="text" placeholder="시나리오명 (비우면 자동)">
      </div>

      <div class="filebox">
        <input id="photo" type="file" accept="image/*" capture="environment">
        <label class="filebtn" for="photo">📷 사진 촬영/선택</label>
      </div>

      <button id="submit" class="primary" disabled>이미지 업로드</button>
      <p id="note" class="sr-only" aria-live="polite"></p>
    </section>
  </main>

  <!-- Bottom tabs (데모) -->
  <nav class="tabs" aria-label="탭">
    <div class="tab active"><span class="ico">🏠</span>홈</div>
    <div class="tab"><span class="ico">🛒</span>샵</div>
    <div class="tab"><span class="ico">🚘</span>제어</div>
    <div class="tab"><span class="ico">🛠️</span>서비스</div>
    <div class="tab"><span class="ico">👤</span>마이</div>
  </nav>

  <div id="toast" class="toast" role="status" aria-live="polite"></div>

<script>
const $ = s=>document.querySelector(s);
const photo=$('#photo'), submit=$('#submit'), toast=$('#toast'), note=$('#note');

function showToast(t){ toast.textContent=t; toast.style.display='block'; setTimeout(()=>toast.style.display='none', 1700); }

photo.addEventListener('change', ()=>{ submit.disabled = !photo.files.length; });

async function pollStatus(scenario){
  try{
    const r = await fetch('/api/status?scenario='+encodeURIComponent(scenario));
    const d = await r.json();
    if(d.ok && d.status === 'done'){ showToast('이미지 분석이 완료되었습니다!'); note.textContent='분석 완료'; return true; }
    if(d.ok && d.status === 'error'){ showToast('분석 실패'); note.textContent='분석 실패'; return true; }
  }catch(e){}
  return false;
}

submit.addEventListener('click', async ()=>{
  if(!photo.files.length) return;
  submit.disabled=true; note.textContent='업로드 중...';

  // 시나리오명(빈 경우 자동)
  let scenario = $('#scenario').value.trim();
  const fd=new FormData();
  if(scenario) fd.append('scenario', scenario);
  fd.append('photo', photo.files[0]);

  try{
    const r=await fetch('/api/upload',{method:'POST',body:fd});
    const d=await r.json();
    if(!r.ok||!d.ok){ showToast('업로드 실패'); note.textContent='업로드 실패'; submit.disabled=false; return; }

    // 1단계: 업로드 완료 토스트
    showToast('이미지가 업로드 되었습니다!');
    note.textContent='업로드 완료';

    scenario = d.scenario; // 서버가 최종 사용한 시나리오명
    // 2단계: 분석 완료까지 폴링
    let tries = 0;
    const timer = setInterval(async ()=>{
      tries += 1;
      const done = await pollStatus(scenario);
      if(done || tries > 120){ clearInterval(timer); submit.disabled=false; photo.value=''; $('#scenario').value=''; }
    }, 1500);

  }catch(e){
    showToast('네트워크 오류'); note.textContent='네트워크 오류'; submit.disabled=false;
  }
});
</script>
</body>
</html>
"""

# ====================== QR (초미니멀) ======================
QR_HTML = """
<!doctype html>
<html lang="ko">
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#000000">
<title>마이현대 앱으로 접속하기</title>
<style>
  html,body{
    height:100%; margin:0;
    background:radial-gradient(1200px 600px at 50% 40%, #0d1117 0%, #0b0e14 45%, #05070b 100%);
    color:#e6edf3; font-family:system-ui,-apple-system,Segoe UI,Roboto,Noto Sans KR,sans-serif
  }
  .wrap{height:100%; display:grid; place-items:center}
  .box{display:flex; flex-direction:column; align-items:center; gap:28px}
  .h{font-weight:900; letter-spacing:.3px; text-align:center; font-size:min(6vw,36px)}
  .qr{width:min(70vmin,520px); height:min(70vmin,520px); background:#fff; border-radius:24px; padding:18px; box-shadow:0 22px 70px rgba(0,0,0,.55)}
</style>
</head>
<body>
  <div class="wrap">
    <div class="box">
      <div class="h">마이현대 앱으로 접속하기</div>
      <img class="qr" src="/qr.png?t={{ts}}" alt="접속 QR">
    </div>
  </div>
<script>
document.addEventListener('click', ()=>{ if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(()=>{}); });
</script>
</body>
</html>
"""

# ====================== 저장 유틸: '원본 그대로' ======================
def _ext_from_filename(filename: str) -> str:
    """원본 파일명에서 확장자 도출(.jpg 등). 없으면 빈 문자열."""
    if not filename:
        return ""
    name = str(filename)
    dot = name.rfind(".")
    return name[dot:].lower() if dot != -1 else ""

def _guess_ext_by_content(raw: bytes, fallback_ext: str) -> str:
    """
    Pillow로 포맷 감지 → 확장자 추정.
    fallback_ext가 있으면 우선 사용, 없으면 감지 결과 사용.
    """
    fmt = None
    try:
        from PIL import Image
        bio = BytesIO(raw)
        with Image.open(bio) as im:
            fmt = (im.format or "").lower()
    except Exception:
        pass

    table = {
        "jpeg": ".jpeg",
        "jpg":  ".jpg",
        "png":  ".png",
        "webp": ".webp",
        "heic": ".heic",
        "heif": ".heic",
        "bmp":  ".bmp",
        "gif":  ".gif",
        "tiff": ".tiff",
    }
    if fallback_ext:
        return fallback_ext if fallback_ext.startswith(".") else "."+fallback_ext
    if fmt in table:
        return table[fmt]
    # 마지막 안전망: 바이너리
    return ".bin"

def _save_upload_as_original(file_storage, scenario: str, base_dir: Path) -> Path:
    """
    업로드된 파일을 **원본 바이트 그대로** 저장한다.
    - 확장자: 원본 파일명 유지, 없으면 포맷 감지로 부여
    - 아무 변환/보정/리사이즈 없음
    반환: 최종 저장 경로(Path)
    """
    raw = file_storage.read()
    orig_ext = _ext_from_filename(file_storage.filename or "")
    final_ext = _guess_ext_by_content(raw, orig_ext)
    dst_path = base_dir / f"{scenario}{final_ext}"
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "wb") as f:
        f.write(raw)  # ✅ 무손실 그대로 저장
    return dst_path

# ====================== 서버 유틸 ======================
def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

def _server_url() -> str:
    return f"http://{_lan_ip()}:{PORT}"

# ====================== 로고/아이콘 ======================
@app.route("/brand.png")
def brand_png():
    """
    brand/hyundai_logo.png 가 있으면 그걸 사용.
    없으면 현대 블루 톤으로 심플한 대체 로고 생성.
    """
    fs = BRAND_DIR / "hyundai_logo.png"
    if fs.exists():
        with open(fs, "rb") as f:
            data = f.read()
        resp = make_response(data); resp.headers["Content-Type"] = "image/png"; return resp
    # 대체 로고 생성
    im = Image.new("RGB", (220, 60), (0, 44, 95))  # #002C5F
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
        d.text((16, 20), "HYUNDAI", fill=(255,255,255), font=font)
    except Exception:
        d.rectangle([18, 18, 202, 42], fill=(255,255,255))
    buf = BytesIO(); im.save(buf, format="PNG")
    resp = make_response(buf.getvalue()); resp.headers["Content-Type"] = "image/png"; return resp

# ====================== QR 생성 ======================
@app.route("/qr.png")
def qr_png():
    url = _server_url()
    if QR_AVAILABLE:
        qr = qrcode.QRCode(box_size=10, border=2)
        qr.add_data(url); qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    else:
        img = Image.new("RGB", (512,512), "white"); d = ImageDraw.Draw(img); d.text((16,240), url, fill="black")
    buf = BytesIO(); img.save(buf, format="PNG")
    resp = make_response(buf.getvalue()); resp.headers["Content-Type"] = "image/png"; return resp

# ====================== 라우트: 홈/QR ======================
@app.route("/", methods=["GET"])
def home():
    return APP_HTML

@app.route("/qr", methods=["GET"])
def qr_fullscreen():
    return render_template_string(QR_HTML, ts=datetime.now().timestamp())

# ====================== 업로드 & 상태 ======================
def _analyze_in_background(scenario: str, img_path: Path):
    """백그라운드에서 분석 실행 후 상태 업데이트."""
    try:
        JOBS[scenario]["status"] = "processing"
        analyzer = LuggageAnalyzer(scenario_name=scenario)
        analyzer.image_path = img_path
        result = analyzer.analyze_image()
        if result:
            analyzer.save_result(result)   # chain1_out/<scenario>.txt
            JOBS[scenario]["status"] = "done"
            JOBS[scenario]["out_txt"] = f"chain1_out/{scenario}.txt"
        else:
            JOBS[scenario]["status"] = "error"
            JOBS[scenario]["error_msg"] = "분석 실패(result 비어 있음)"
    except Exception as e:
        JOBS[scenario]["status"] = "error"
        JOBS[scenario]["error_msg"] = str(e)

@app.route("/api/upload", methods=["POST"])
def api_upload():
    scenario = (request.form.get("scenario") or "").strip()
    if not scenario:
        scenario = datetime.now().strftime("items_%Y%m%d_%H%M%S")

    file = request.files.get("photo")
    if not file:
        return jsonify(ok=False, error="파일이 없습니다."), 400

    try:
        # ✅ 원본 그대로 저장 (확장자 유지)
        img_path = _save_upload_as_original(file, scenario, IMG_DIR)

        # 상태 등록 + 백그라운드 분석 시작
        JOBS[scenario] = {"status": "uploaded", "out_txt": None, "error_msg": None}
        th = threading.Thread(target=_analyze_in_background, args=(scenario, img_path), daemon=True)
        th.start()

        # 업로드 즉시 응답
        return jsonify(ok=True, stage="uploaded", scenario=scenario)
    except Exception as e:
        return jsonify(ok=False, error=f"오류: {e}"), 500

@app.route("/api/status", methods=["GET"])
def api_status():
    scenario = (request.args.get("scenario") or "").strip()
    if not scenario or scenario not in JOBS:
        return jsonify(ok=False, error="unknown scenario"), 404
    return jsonify(ok=True, status=JOBS[scenario]["status"], out_txt=JOBS[scenario]["out_txt"], error_msg=JOBS[scenario]["error_msg"])

# ====================== Entrypoint ======================
if __name__ == "__main__":
    url = _server_url()
    print(f"\n휴대폰에서 접속:  {url}\n(같은 Wi-Fi 필요)\n")
    if AUTO_OPEN_BROWSER:
        try:
            import webbrowser
            webbrowser.open(f"http://127.0.0.1:{PORT}{AUTO_OPEN_BROWSER_PATH}")
        except Exception:
            pass
    app.run(host="0.0.0.0", port=PORT)
