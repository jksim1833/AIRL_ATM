# test_langsmith_headers.py
import os, requests, json
from pathlib import Path
from dotenv import load_dotenv

p = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=p, override=True)

key = os.getenv("LANGSMITH_API_KEY")
proj = os.getenv("LANGCHAIN_PROJECT")
url_base = os.getenv("LANGSMITH_API_URL") or "https://api.smith.langchain.com"
url = url_base.rstrip("/") + "/runs"

print("Using .env:", p.exists())
print("PROJECT:", proj)
print("URL:", url)
print("raw key repr:", repr(key))
print("len:", len(key) if key else None)

payload = {"run_type":"chain","project_name":proj,"name":"test-run-dbg","inputs":{},"tags":["dbg"]}

def try_headers(headers, label):
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        print(f"\n--- {label} ---")
        print("STATUS:", r.status_code)
        print("BODY:", r.text)
    except Exception as e:
        print(f"\n--- {label} EXCEPTION ---")
        print(repr(e))

# 1) Authorization: Bearer
try_headers({"Authorization": f"Bearer {key}", "Content-Type":"application/json"}, "Authorization: Bearer")

# 2) x-api-key (lowercase)
try_headers({"x-api-key": key, "Content-Type":"application/json"}, "x-api-key (lowercase)")

# 3) X-Api-Key (common variant)
try_headers({"X-Api-Key": key, "Content-Type":"application/json"}, "X-Api-Key (capitalized)")

# 4) Both headers (just to test)
try_headers({"Authorization": f"Bearer {key}", "x-api-key": key, "Content-Type":"application/json"}, "Both headers")
