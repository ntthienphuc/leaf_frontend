$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath ".\.venv")) {
    py -3.11 -m venv .venv --system-site-packages
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
