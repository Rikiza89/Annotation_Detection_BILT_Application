@echo off
setlocal enabledelayedexpansion

REM Configuration
set VENV_PATH="%~dp0..\venv\Scripts\activate.bat"
set VENV_PYTHON="%~dp0..\venv\Scripts\python.exe"
set CLIENT_APP="%~dp0bilt_service.py"
set GUI_APP="%~dp0annotation_app.py"
set URL=http://127.0.0.1:5000
set SETTINGS_FILE="%~dp0flask_venv_settings.txt"
set PID_FILE="%~dp0flask_venv.pid"
set CLIENT_PID_FILE="%~dp0client_service.pid"

REM Load settings or set defaults
if exist %SETTINGS_FILE% (
    for /f "tokens=1,2 delims==" %%a in (%SETTINGS_FILE%) do (
        if "%%a"=="SHOW_WINDOW" set SHOW_WINDOW=%%b
    )
) else (
    set SHOW_WINDOW=1
)

REM Main Menu
:main_menu
cls
echo ************************************************
echo ================================================
echo ************************************************
echo    COMBINED ANNOTATION SYSTEM MANAGER (VENV)
echo ************************************************
echo ================================================
echo ************************************************
echo.
echo Current Settings:
echo   Python Window: !SHOW_WINDOW! (1=Show, 0=Hide)

REM Check if Client Service is running
set CLIENT_RUNNING=0
if exist %CLIENT_PID_FILE% (
    set CLIENT_RUNNING=1
    echo   Client Service: RUNNING
) else (
    echo   Client Service: STOPPED
)

REM Check if Flask GUI is running
set SERVER_RUNNING=0
curl -s %URL% >nul 2>&1
if %errorlevel% equ 0 (
    set SERVER_RUNNING=1
    echo   GUI Server: RUNNING
) else (
    echo   GUI Server: STOPPED
    REM Clean up stale PID file if server is not responding
    if exist %PID_FILE% del %PID_FILE% 2>nul
)

echo.
echo ================================================
echo.
echo [1] Start Complete System (Client + GUI)
echo [2] Start Client Service Only
echo [3] Start GUI Server Only
echo [4] Stop Client Service
echo [5] Stop GUI Server
echo [6] Stop All Services
echo [7] Toggle Python Window (Currently: !SHOW_WINDOW!)
echo [8] Open Browser
echo [9] View System Status
echo [0] Exit (Auto-stops all services)
echo.
set /p choice="Select an option (0-9): "

if "%choice%"=="1" goto start_complete
if "%choice%"=="2" goto start_client
if "%choice%"=="3" goto start_gui
if "%choice%"=="4" goto stop_client
if "%choice%"=="5" goto stop_gui
if "%choice%"=="6" goto stop_all
if "%choice%"=="7" goto toggle_window
if "%choice%"=="8" goto open_browser
if "%choice%"=="9" goto check_status
if "%choice%"=="0" goto exit_script
echo Invalid choice! Please try again.
timeout /t 2 >nul
goto main_menu

:start_complete
cls
echo ================================================
echo      STARTING COMPLETE ANNOTATION SYSTEM
echo ================================================
echo.

REM Start Client Service
call :start_client_internal

REM Wait a moment
echo Waiting for client service to initialize...
timeout /t 5 /nobreak >nul

REM Start GUI Server
call :start_gui_internal

pause
goto main_menu

:start_client
cls
echo ================================================
echo         STARTING CLIENT SERVICE
echo ================================================
echo.
call :start_client_internal
pause
goto main_menu

:start_client_internal
REM Check if already running
if exist %CLIENT_PID_FILE% (
    echo Client service is already running!
    goto :eof
)

REM Validate paths
if not exist %VENV_PATH% (
    echo ERROR: Virtual environment not found at %VENV_PATH%
    goto :eof
)

if not exist %CLIENT_APP% (
    echo ERROR: bilt_service.py not found at %CLIENT_APP%
    goto :eof
)

echo Activating virtual environment...
call %VENV_PATH%

echo Starting client service in background...
if "!SHOW_WINDOW!"=="1" (
    start "Client Service (venv)" cmd /k "call %VENV_PATH% && python %CLIENT_APP%"
) else (
    start "Client Service (venv)" /min cmd /k "call %VENV_PATH% && python %CLIENT_APP%"
)

REM Mark as running
echo Started > %CLIENT_PID_FILE%
echo Client service started successfully!
goto :eof

:start_gui
cls
echo ================================================
echo         STARTING GUI SERVER
echo ================================================
echo.
call :start_gui_internal
pause
goto main_menu

:start_gui_internal
REM Check if actually running by testing the URL
curl -s %URL% >nul 2>&1
if %errorlevel% equ 0 (
    echo ERROR: GUI server is already running!
    echo Server is responding at %URL%
    goto :eof
)

REM Clean up any stale PID file
if exist %PID_FILE% (
    echo Cleaning up stale PID file...
    del %PID_FILE% 2>nul
)

REM Validate paths
if not exist %VENV_PATH% (
    echo ERROR: Virtual environment not found at %VENV_PATH%
    goto :eof
)

if not exist %GUI_APP% (
    echo ERROR: annotation_app.py not found at %GUI_APP%
    goto :eof
)

echo Activating virtual environment...
call %VENV_PATH%

echo Starting GUI server...
if "!SHOW_WINDOW!"=="1" (
    start "GUI Server (venv)" cmd /k "call %VENV_PATH% && python %GUI_APP%"
) else (
    start "GUI Server (venv)" /min cmd /k "call %VENV_PATH% && python %GUI_APP%"
)

REM Mark as running
echo Started > %PID_FILE%

echo Waiting for GUI server to initialize...
set /a attempts=0
:wait_gui_loop
timeout /t 2 /nobreak >nul
set /a attempts+=1
echo Checking if server is ready... (Attempt %attempts%/15)
curl -s %URL% >nul 2>&1
if %errorlevel% equ 0 (
    echo GUI server is ready and running!
    goto :eof
)
if %attempts% geq 15 (
    echo WARNING: Server took too long to start.
    del %PID_FILE% 2>nul
    goto :eof
)
goto wait_gui_loop

:stop_client
cls
echo ================================================
echo         STOPPING CLIENT SERVICE
echo ================================================
echo.

if not exist %CLIENT_PID_FILE% (
    echo Client service is not running.
    pause
    goto main_menu
)

echo Stopping client service...
taskkill /F /FI "WINDOWTITLE eq Client Service (venv)" 2>nul
if %errorlevel% neq 0 (
    taskkill /F /FI "WINDOWTITLE eq Client Service (venv)*" 2>nul
)

del %CLIENT_PID_FILE% 2>nul
echo Client service stopped successfully!
pause
goto main_menu

:stop_gui
cls
echo ================================================
echo         STOPPING GUI SERVER
echo ================================================
echo.

curl -s %URL% >nul 2>&1
if %errorlevel% neq 0 (
    echo GUI server is not running.
    if exist %PID_FILE% del %PID_FILE% 2>nul
    pause
    goto main_menu
)

echo Stopping GUI server...
taskkill /F /FI "WINDOWTITLE eq GUI Server (venv)" 2>nul
if %errorlevel% neq 0 (
    taskkill /F /FI "WINDOWTITLE eq GUI Server (venv)*" 2>nul
)

REM Verify and force kill if needed
timeout /t 1 /nobreak >nul
curl -s %URL% >nul 2>&1
if %errorlevel% equ 0 (
    echo Trying port-based kill...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING') do (
        taskkill /F /PID %%a /T 2>nul
    )
)

del %PID_FILE% 2>nul
echo GUI server stopped successfully!
pause
goto main_menu

:stop_all
cls
echo ================================================
echo         STOPPING ALL SERVICES
echo ================================================
echo.

REM Stop Client
if exist %CLIENT_PID_FILE% (
    echo Stopping client service...
    taskkill /F /FI "WINDOWTITLE eq Client Service (venv)*" 2>nul
    del %CLIENT_PID_FILE% 2>nul
    echo Client service stopped.
)

REM Stop GUI
curl -s %URL% >nul 2>&1
if %errorlevel% equ 0 (
    echo Stopping GUI server...
    taskkill /F /FI "WINDOWTITLE eq GUI Server (venv)*" 2>nul
    timeout /t 1 /nobreak >nul
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING') do (
        taskkill /F /PID %%a /T 2>nul
    )
    del %PID_FILE% 2>nul
    echo GUI server stopped.
)

echo All services stopped successfully!
pause
goto main_menu

:toggle_window
if "!SHOW_WINDOW!"=="1" (
    set SHOW_WINDOW=0
    echo Python windows will be HIDDEN on next start.
) else (
    set SHOW_WINDOW=1
    echo Python windows will be VISIBLE on next start.
)

REM Save settings
echo SHOW_WINDOW=!SHOW_WINDOW! > %SETTINGS_FILE%
timeout /t 2 >nul
goto main_menu

:open_browser
cls
echo Opening browser at %URL%...
start "" "%URL%"
timeout /t 2 >nul
goto main_menu

:check_status
cls
echo ================================================
echo          SYSTEM STATUS CHECK
echo ================================================
echo.
echo Client Service:
if exist %CLIENT_PID_FILE% (
    echo   Status: RUNNING (PID file exists)
) else (
    echo   Status: STOPPED
)
echo.
echo GUI Server:
curl -s %URL% >nul 2>&1
if %errorlevel% equ 0 (
    echo   Status: RUNNING (responding at %URL%)
) else (
    echo   Status: STOPPED (not responding)
)
if exist %PID_FILE% (
    echo   PID File: EXISTS
) else (
    echo   PID File: NOT FOUND
)
echo.
echo Virtual Environment:
if exist %VENV_PATH% (
    echo   Found at ..\venv\Scripts\activate.bat
) else (
    echo   NOT FOUND - Please create venv first!
)
echo.
pause
goto main_menu

:exit_script
cls
echo ================================================
echo          EXITING ANNOTATION SYSTEM
echo ================================================
echo.

REM Stop all services
if exist %CLIENT_PID_FILE% (
    echo Stopping client service...
    taskkill /F /FI "WINDOWTITLE eq Client Service (venv)*" 2>nul
    del %CLIENT_PID_FILE% 2>nul
)

curl -s %URL% >nul 2>&1
if %errorlevel% equ 0 (
    echo Stopping GUI server...
    taskkill /F /FI "WINDOWTITLE eq GUI Server (venv)*" 2>nul
    timeout /t 1 /nobreak >nul
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING') do (
        taskkill /F /PID %%a /T 2>nul
    )
    del %PID_FILE% 2>nul
)

echo All services stopped.
echo Goodbye!
timeout /t 2 >nul
exit /b