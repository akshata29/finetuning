@echo off
REM Start the demo API. The package "finetuning" must be importable, so we
REM run from the repo root (parent of this folder) and point python at the venv.
setlocal
set "DEMO_DIR=%~dp0"
pushd "%DEMO_DIR%.."
"%DEMO_DIR%.venv\Scripts\python.exe" -m finetuning.api %*
popd
endlocal