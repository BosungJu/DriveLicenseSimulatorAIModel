@echo off
REM vexp-hint: event-driven orientation hint (UserPromptSubmit). Fails open.
set "VEXP_BIN=C:\Users\joobs\.vscode\extensions\vexp.vexp-vscode-3.3.1-win32-x64\binaries\vexp-core-win32-x64\vexp-core.exe"
if not exist "%VEXP_BIN%" exit /b 0
set "VEXP_HOOK_AGENT=codex"
"%VEXP_BIN%" prompt-hint 2>nul
exit /b 0
