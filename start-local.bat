@echo off
REM Start FedCare servers locally (Windows)

setlocal enabledelayedexpansion

echo ========================================
echo FedCare - Starting Servers Locally
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed
    exit /b 1
)

REM Check if venv exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing dependencies...
pip install -q -r requirements.txt

REM Create logs directory
if not exist "logs" mkdir logs

echo.
echo Starting servers...
echo.

REM Start main server
echo Starting Main Server on port 5000...
start "FedCare Main Server" python main_server/app.py
echo.

REM Wait a bit for main server
timeout /t 2 /nobreak

REM Start hospital servers
echo Starting Hospital Server 1 on port 5001...
start "FedCare Hospital 1" cmd /c "set HOSPITAL_ID=hospital_1 && set PORT=5001 && python hospital_server/app.py"
echo.

echo Starting Hospital Server 2 on port 5002...
start "FedCare Hospital 2" cmd /c "set HOSPITAL_ID=hospital_2 && set PORT=5002 && python hospital_server/app.py"
echo.

echo Starting Hospital Server 3 on port 5003...
start "FedCare Hospital 3" cmd /c "set HOSPITAL_ID=hospital_3 && set PORT=5003 && python hospital_server/app.py"
echo.

echo ========================================
echo Servers started successfully!
echo ========================================
echo.
echo Main Server: http://localhost:5000
echo Hospital 1:  http://localhost:5001
echo Hospital 2:  http://localhost:5002
echo Hospital 3:  http://localhost:5003
echo.
echo You can now run the examples or orchestrator:
echo   python examples.py
echo   python orchestrator.py
echo.
echo Press Ctrl+C in any window to stop a server
echo.

pause
