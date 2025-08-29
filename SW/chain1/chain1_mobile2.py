# -*- coding: utf-8 -*-
"""
chain1_mobile2.py
- 모바일: 사진 촬영 → 리뷰(원본 미리보기) → 업로드 → 상태 폴링
- 저장 규칙(원본 그대로):
    * chain1_image/: <scenario>.<원본확장자>
    * chain1_out/  : <scenario>.txt
- UI:
    * 상단 히어로 하단만 '은은한 블러 + 화이트 페이드'
    * 탑승 인원 칩: 선택 시 남색(#002c5f) 반전
    * 리뷰 모달: '다시 촬영' / '사진 업로드'(원본 변형 없음)
    * 업로드 성공 후 하단 네모 배너 2단계 + '최적 배치 시작' 활성화(남색)
- /brand/<file> 정적 서빙(폰트/히어로 이미지)
- /qr: 풀스크린 QR + 남색 라벨(“마이현대 앱에서 AI TETRIS를 경험해보세요!”)

필요:
    pip install flask pillow pillow-heif qrcode[pil]
"""

from pathlib import Path
from datetime import datetime
from io import BytesIO
from flask import Flask, request, jsonify, make_response, render_template_string, send_from_directory, abort
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

# woff2 MIME 보정
mimetypes.add_type("font/woff2", ".woff2")

# ====================== 프론트 (최종 UI) ======================
APP_HTML = """
<!doctype html>
<html lang="ko">
<head>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#ffffff">
<title>AI TETRIS</title>
<style>
  /* ===== 폰트 ===== */
  @font-face{
    font-family:"HyundaiHarmonyM";
    src:url("/brand/HyundaiHarmonyM.woff2") format("woff2");
    font-weight:400; font-style:normal; font-display:swap;
  }
  @font-face{
    font-family:"HyundaiHarmonyL";
    src:url("/brand/HyundaiHarmonyL.woff2") format("woff2");
    font-weight:300; font-style:normal; font-display:swap;
  }

  :root{
    --navy:#002c5f;
    --black:#000;
    --white:#fff;
    --dark:#374151;
    --light:#E5E7EB;
    --chipbg:#F3F4F6;
  }

  *{box-sizing:border-box}
  html,body{
    margin:0; background:#fff; color:var(--black);
    font-family:"HyundaiHarmonyM","Noto Sans KR",system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    -webkit-font-smoothing:antialiased;
  }

  /* ===== 상단 바 ===== */
  .topbar{
    position:sticky; top:0; z-index:20; background:#fff;
    display:flex; align-items:center; justify-content:center;
    padding:14px 16px; border-bottom:1px solid var(--light);
  }
  .topbar h1{ margin:0; font-size:18px; color:var(--black); letter-spacing:.2px; }
  .topbar .close{ position:absolute; right:16px; top:50%; transform:translateY(-50%);
    font-size:18px; color:var(--black); text-decoration:none; font-weight:700; }

  /* ===== 히어로 (하단만 블러 + 화이트 페이드) ===== */
  .hero{
    position:relative; width:100%; height:220px; overflow:hidden; background:#0f172a;
  }
  .hero img{ width:100%; height:100%; object-fit:cover; display:block; }
  /* 하단 영역만 살짝 블러하고, 위로 갈수록 투명해지는 마스크를 얹는다 */
  .hero-blur{
    position:absolute; left:0; right:0; bottom:0; height:36%;
    backdrop-filter: blur(10px) saturate(1.05);
    -webkit-backdrop-filter: blur(10px) saturate(1.05);
    pointer-events:none;
  }
  .hero-blur::before{
    content:""; position:absolute; inset:0;
    background:linear-gradient(0deg, rgba(255,255,255,1) 0%,
                                     rgba(255,255,255,.85) 35%,
                                     rgba(255,255,255,0) 100%);
  }
  .hero-text{
    position:absolute; left:16px; bottom:18px; color:var(--white);
    text-shadow:0 2px 8px rgba(0,0,0,.35); max-width:82%;
  }
  .hero-title{ margin:0 0 6px 0; font-size:24px; line-height:1.2; font-family:"HyundaiHarmonyM"; }
  .hero-sub{ margin:0; font-size:13px; line-height:1.4; font-family:"HyundaiHarmonyL"; }

  .wrap{ padding:16px; }

  /* ===== 섹션 헤더 ===== */
  .section-head{ display:flex; align-items:center; gap:8px; margin:16px 0 8px 0; }
  .section-head svg{ width:18px; height:18px; }
  .section-title{ margin:0; font-size:16px; color:var(--black); font-family:"HyundaiHarmonyM"; }
  .section-desc{ margin:6px 0 0 26px; font-size:12px; color:var(--dark); line-height:1.5; font-family:"HyundaiHarmonyL"; }

  /* ===== 탑승 인원 칩 ===== */
  .chips{ display:grid; grid-template-columns:repeat(2, 1fr); gap:10px; margin-top:12px; }
  .chip{
    display:flex; align-items:center; justify-content:center;
    height:44px; border-radius:22px;
    background:var(--chipbg);
    border:2px solid var(--navy);
    color:var(--navy); font-size:16px; font-weight:700;
    text-decoration:none; user-select:none; cursor:pointer;
  }
  .chip.selected{ background:var(--navy); color:#fff; }

  .divider{ height:1px; background:var(--light); margin:18px 0; }

  /* ===== 사진촬영 박스 ===== */
  .photo-box{ border:2px solid var(--light); border-radius:16px; padding:16px;
    display:grid; place-items:center; min-height:180px; }
  .btn-photo{
    display:inline-flex; align-items:center; justify-content:center;
    min-width:160px; height:44px; padding:0 18px;
    border-radius:12px; border:2px solid var(--navy);
    background:var(--navy); color:var(--white);
    font-size:15px; font-weight:800; text-decoration:none; cursor:pointer;
  }
  #photo{ position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); border:0; }

  /* ===== 메인 CTA (활성/비활성) ===== */
  .cta{
    margin-top:18px; width:100%; height:50px; border-radius:14px;
    border:2px solid #cbd5e1; background:#e5e7eb; color:#475569;
    font-size:16px; font-weight:900;
  }
  .cta.active{ border-color:var(--navy); background:var(--navy); color:#fff; }

  .sr-only{ position:absolute; left:-10000px; }

  /* ===== 촬영 후 리뷰 모달 ===== */
  .review{ position:fixed; inset:0; z-index:50; display:none; background:rgba(0,0,0,.55);
    align-items:center; justify-content:center; padding:16px; }
  .review.show{ display:flex; }
  .review-sheet{
    width:min(420px, 96vw); max-height:90vh; background:#fff; border-radius:24px;
    box-shadow:0 20px 60px rgba(0,0,0,.35); overflow:hidden; display:flex; flex-direction:column;
    font-family:"HyundaiHarmonyM";
  }
  .review-image{ background:#000; display:grid; place-items:center; padding:0; }
  .review-image img{
    display:block; max-width:100%; max-height:60vh; height:auto; width:auto;
    image-rendering:auto; image-orientation:from-image;
  }
  .review-actions{ padding:16px; display:flex; flex-direction:column; gap:10px; }
  .review-actions .btn{
    height:48px; border-radius:12px; border:2px solid var(--navy);
    background:var(--navy); color:#fff; font-size:17px; font-weight:900;
    display:flex; align-items:center; justify-content:center; cursor:pointer;
  }

  /* ===== 하단 네모 배너(알림) ===== */
  .notice{
    position:fixed; left:50%; bottom:18px; transform:translateX(-50%);
    padding:10px 14px; border-radius:12px; background:#fff; color:#0b1220;
    border:2px solid var(--navy); box-shadow:0 10px 26px rgba(12,18,32,.12);
    font-size:14px; display:none; z-index:60;
  }
  .notice.show{ display:block; }
</style>
</head>
<body>

  <!-- 상단바 -->
  <div class="topbar">
    <h1>AI TETRIS</h1>
    <a class="close" href="#" aria-label="닫기">X</a>
  </div>

  <!-- 히어로 -->
  <section class="hero" aria-hidden="false">
    <img src="/brand/AI_TETRIS.png" alt="AI TETRIS">
    <div class="hero-blur" aria-hidden="true"></div>
    <div class="hero-text">
      <h2 class="hero-title">현대자동차 AI TETRIS 서비스</h2>
      <p class="hero-sub">단 한 장의 짐 사진으로 최적의 차량 시트 배치를<br>자동으로 완성하는 서비스입니다.</p>
    </div>
  </section>

  <main class="wrap">
    <!-- 탑승 인원 -->
    <div class="section-head">
      <!-- 사람 아이콘: 내장 SVG -->
      <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="8" r="4" fill="#9ca3af"/><rect x="5" y="13" width="14" height="8" rx="4" fill="#9ca3af"/></svg>
      <h3 class="section-title">탑승 인원</h3>
    </div>
    <p class="section-desc">1열을 제외한 차량 탑승 인원을 알려주세요</p>

    <div class="chips" role="group" aria-label="탑승 인원 선택">
      <button class="chip" type="button" data-seats="1">1명</button>
      <button class="chip" type="button" data-seats="2">2명</button>
      <button class="chip" type="button" data-seats="3">3명</button>
      <button class="chip" type="button" data-seats="4">4명</button>
    </div>

    <div class="divider" aria-hidden="true"></div>

    <!-- 짐 -->
    <div class="section-head" style="margin-top:10px">
      <!-- 박스 아이콘: 내장 SVG -->
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 7l9-4 9 4-9 4-9-4zm0 4l9 4 9-4v7l-9 4-9-4v-7z" fill="#9ca3af"/></svg>
      <h3 class="section-title">짐</h3>
    </div>
    <p class="section-desc">대각선 방향에서 짐 이외의 배경은 최대한 보이지 않도록 촬영해주세요<br>개별 짐이 가려지는 부분이 없도록 주의하세요!</p>

    <div class="photo-box" aria-label="사진 촬영 영역">
      <input id="photo" type="file" accept="image/*" capture="environment">
      <label class="btn-photo" for="photo">사진 촬영</label>
    </div>

    <button id="submit" class="cta" disabled>최적 배치 시작</button>
    <p id="note" class="sr-only" aria-live="polite"></p>
  </main>

  <!-- 촬영 후 리뷰 -->
  <div id="review" class="review" aria-hidden="true" role="dialog" aria-modal="true">
    <div class="review-sheet">
      <div class="review-image">
        <img id="review-img" alt="촬영 이미지 미리보기">
      </div>
      <div class="review-actions">
        <button id="btn-retake" class="btn">다시 촬영</button>
        <button id="btn-send" class="btn">사진 업로드</button>
      </div>
    </div>
  </div>

  <!-- 하단 네모 배너 -->
  <div id="notice" class="notice" role="status" aria-live="polite"></div>

<script>
const $ = s=>document.querySelector(s);

/* 탑승 인원 선택 → 선택 칩 남색 */
const chips = document.querySelectorAll('.chip');
let seatSelection = null;
chips.forEach(ch=>{
  ch.addEventListener('click', ()=>{
    chips.forEach(c=>c.classList.remove('selected'));
    ch.classList.add('selected');
    seatSelection = ch.dataset.seats; // 필요 시 서버 전송에 사용 가능
  });
});

/* 파일/리뷰/업로드 로직 */
const photo=$('#photo'), submit=$('#submit'), note=$('#note');
const review=$('#review'), reviewImg=$('#review-img');
const btnRetake=$('#btn-retake'), btnSend=$('#btn-send');
const notice=$('#notice');

let previewURL=null, currentScenario=null, pollTimer=null;

function showNotice(msg, duration=1400){
  notice.textContent = msg;
  notice.classList.add('show');
  setTimeout(()=> notice.classList.remove('show'), duration);
}

/* 촬영되면 리뷰 모달로 열기(원본 표시) */
photo.addEventListener('change', ()=>{
  if(!photo.files.length) return;
  if(previewURL) URL.revokeObjectURL(previewURL);
  previewURL = URL.createObjectURL(photo.files[0]); // 원본 그대로 표기(화면 맞춤 축소만)
  reviewImg.src = previewURL;
  review.classList.add('show');
});

/* 다시 촬영 */
btnRetake.addEventListener('click', ()=>{
  review.classList.remove('show');
  if(previewURL) { URL.revokeObjectURL(previewURL); previewURL=null; }
  photo.value='';
  setTimeout(()=>{ photo.click(); }, 50); // 카메라 재오픈 유도
});

/* 사진 업로드 → 하단 배너 2단계 + CTA 활성화 */
btnSend.addEventListener('click', async ()=>{
  if(!photo.files.length) return;
  btnSend.disabled = true;

  try{
    const fd=new FormData();
    fd.append('photo', photo.files[0]);
    // seatSelection 필요 시 함께 전송 가능:
    // if(seatSelection) fd.append('seats', seatSelection);

    const r=await fetch('/api/upload',{method:'POST',body:fd});
    const d=await r.json();
    if(!r.ok||!d.ok){
      alert('업로드 실패');
      btnSend.disabled=false; return;
    }

    currentScenario = d.scenario;
    review.classList.remove('show');

    // 1) 업로드 완료 배너
    showNotice('사진이 업로드 되었습니다');

    // CTA 활성화(남색으로 변경)
    submit.disabled = false;
    submit.classList.add('active');

    // 2) 이어서 두 번째 배너
    setTimeout(()=> showNotice('최적의 차량 배치 구상을 시작합니다'), 1600);

    // 상태 폴링(분석 완료 대기)
    let tries=0;
    clearInterval(pollTimer);
    pollTimer = setInterval(async ()=>{
      tries += 1;
      try{
        const rs = await fetch('/api/status?scenario='+encodeURIComponent(currentScenario));
        const dj = await rs.json();
        if(dj.ok && dj.status === 'done'){
          clearInterval(pollTimer);
          // 필요 시 결과 화면 전환/링크 추가 가능
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

  }catch(e){
    alert('네트워크 오류');
  }finally{
    btnSend.disabled=false;
  }
});
</script>
</body>
</html>
"""

# ====================== QR (업데이트: 남색 라벨 + 문구 변경) ======================
QR_HTML = """
<!doctype html>
<html lang="ko">
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#000000">
<title>QR · AI TETRIS</title>
<style>
  @font-face{
    font-family:"HyundaiHarmonyM";
    src:url("/brand/HyundaiHarmonyM.woff2") format("woff2");
    font-weight:400; font-style:normal; font-display:swap;
  }
  html,body{
    height:100%; margin:0;
    background:radial-gradient(1200px 600px at 50% 40%, #0d1117 0%, #0b0e14 45%, #05070b 100%);
    color:#e6edf3; font-family:"HyundaiHarmonyM",system-ui,-apple-system,Segoe UI,Roboto,Noto Sans KR,sans-serif
  }
  .wrap{height:100%; display:grid; place-items:center; padding:18px}
  .box{display:flex; flex-direction:column; align-items:center; gap:22px}
  .qr{width:min(70vmin,520px); height:min(70vmin,520px); background:#fff; border-radius:24px; padding:18px; box-shadow:0 22px 70px rgba(0,0,0,.55)}
  .label{
    background:#002c5f; color:#fff; padding:12px 16px; border-radius:14px;
    font-size:min(4.6vw,20px); text-align:center; box-shadow:0 12px 30px rgba(0,0,0,.35);
  }
</style>
</head>
<body>
  <div class="wrap">
    <div class="box">
      <img class="qr" src="/qr.png?t={{ts}}" alt="접속 QR">
      <div class="label">마이현대 앱에서 AI TETRIS를 경험해보세요!</div>
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

# ====================== 브랜드/정적 자산 ======================
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

# brand 폴더 서빙 (폰트/이미지)
@app.route("/brand/<path:filename>")
def brand_assets(filename):
    p = BRAND_DIR / filename
    if not p.exists():
        abort(404)
    return send_from_directory(BRAND_DIR, filename)

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
def _lan_url():
    return f"http://{_lan_ip()}:{PORT}"

if __name__ == "__main__":
    url = _lan_url()
    print(f"\n휴대폰에서 접속:  {url}\n(같은 Wi-Fi 필요)\n")
    if AUTO_OPEN_BROWSER:
        try:
            import webbrowser
            webbrowser.open(f"http://127.0.0.1:{PORT}{AUTO_OPEN_BROWSER_PATH}")
        except Exception:
            pass
    app.run(host="0.0.0.0", port=PORT)
