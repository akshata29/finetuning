@echo off
REM Start the Vite dev server for the demo UI.
setlocal
cd /d "%~dp0ui"
if not exist node_modules (
  echo Installing UI dependencies ^(first run^)...
  call npm install
)
call npm run dev
endlocal