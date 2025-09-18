@echo off
setlocal

REM Change to this script's directory
cd /d "%~dp0"

REM Create venv if missing
if not exist ".venv" (
  py -3 -m venv .venv
)

REM Activate venv
call .venv\Scripts\activate

REM Upgrade pip (optional but helpful)
python -m pip install --upgrade pip

REM Install deps
pip install -r requirements.txt

REM Run server on port 8000
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

endlocal
