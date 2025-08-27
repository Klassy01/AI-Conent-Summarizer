@echo off
echo ========================================
echo AI Multi-Source Summarizer Setup
echo ========================================
echo.

echo Step 1: Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    echo Make sure Python 3.8+ is installed
    pause
    exit /b 1
)

echo Step 2: Activating virtual environment...
call venv\Scripts\activate

echo Step 3: Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete! 
echo ========================================
echo.
echo Next steps:
echo 1. Copy .env.example to .env
echo 2. Add your Google API key to .env
echo 3. Run: streamlit run app.py
echo.
echo Get API key at: https://makersuite.google.com/app/apikey
echo.
pause
