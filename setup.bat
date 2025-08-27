@echo off
echo.
echo ===============================================
echo   AI Multi-Source Summarizer - Setup Script
echo ===============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version

echo.
echo 📦 Installing required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ❌ Error installing packages. Please check the error above.
    pause
    exit /b 1
)

echo.
echo ✅ Packages installed successfully!

REM Check if .env file exists
if not exist ".env" (
    echo.
    echo ⚙️ Creating .env file from template...
    copy ".env.example" ".env" >nul
    if errorlevel 1 (
        echo ❌ Could not create .env file
    ) else (
        echo ✅ Created .env file
    )
)

echo.
echo ===============================================
echo   🎉 Setup Complete!
echo ===============================================
echo.
echo 📋 Next Steps:
echo.
echo 1. Get your Google API key:
echo    👉 https://makersuite.google.com/app/apikey
echo.
echo 2. Edit the .env file and add your API key:
echo    GOOGLE_API_KEY=your_actual_api_key_here
echo.
echo 3. Run the application:
echo    streamlit run app.py
echo.
echo 4. Open http://localhost:8501 in your browser
echo.

pause
