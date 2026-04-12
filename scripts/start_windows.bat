@echo off
title LazyMoE Launcher

echo.
echo  ██╗      █████╗ ███████╗██╗   ██╗      ███╗   ███╗ ██████╗ ███████╗
echo  ██║     ██╔══██╗╚════██║╚██╗ ██╔╝      ████╗ ████║██╔═══██╗██╔════╝
echo  ██║     ███████║    ██╔╝ ╚████╔╝ █████╗██╔████╔██║██║   ██║█████╗
echo  ██║     ██╔══██║   ██╔╝   ╚██╔╝  ╚════╝██║╚██╔╝██║██║   ██║██╔══╝
echo  ███████╗██║  ██║   ██║     ██║         ██║ ╚═╝ ██║╚██████╔╝███████╗
echo  ╚══════╝╚═╝  ╚═╝   ╚═╝     ╚═╝         ╚═╝     ╚═╝ ╚═════╝ ╚══════╝
echo.
echo  Local LLM Inference  ^|  LazyMoE v0.3
echo  -------------------------------------------------------

:: Set model path - change this to your model file
set LAZY_MOE_MODEL=%USERPROFILE%\lazy-moe\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf
set LAZY_MOE_RAM_GB=8
set LAZY_MOE_THREADS=4

:: Add llama.cpp to PATH
set PATH=%USERPROFILE%\lazy-moe\llama.cpp;%PATH%

echo  [1/2] Starting backend server...
start "LazyMoE Backend" cmd /k "cd /d %USERPROFILE%\lazy-moe\backend && python server.py"

echo  [2/2] Waiting 3 seconds then starting frontend...
timeout /t 3 /nobreak > nul

start "LazyMoE Frontend" cmd /k "cd /d %USERPROFILE%\lazy-moe\frontend && npm run dev"

echo.
echo  Waiting for servers to start...
timeout /t 5 /nobreak > nul

echo.
echo  Opening browser...
start http://localhost:5173

echo.
echo  LazyMoE is running!
echo  Backend:  http://localhost:8000
echo  Frontend: http://localhost:5173
echo.
echo  Close this window when done (it will keep running in background windows)
pause
