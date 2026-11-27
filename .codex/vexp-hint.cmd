@echo off
REM vexp-hint: event-driven orientation hint (UserPromptSubmit). Fails open.
rem vexp-engine: 1 - the engine through vexp's version-free link. Managed by vexp.
set "VEXP_BIN="
set "_vexp_n=VEXP_BIN"
if defined VEXP_HOME if exist "%VEXP_HOME%\.vexp\bin\vexp-core.exe" set "%_vexp_n%=%VEXP_HOME%\.vexp\bin\vexp-core.exe"
if not defined VEXP_BIN if exist "%USERPROFILE%\.vexp\bin\vexp-core.exe" set "%_vexp_n%=%USERPROFILE%\.vexp\bin\vexp-core.exe"
if not exist "%VEXP_BIN%" exit /b 0
set "VEXP_HOOK_AGENT=codex"
"%VEXP_BIN%" prompt-hint 2>nul
exit /b 0
