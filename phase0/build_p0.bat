@echo off
cd /d %~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cl /nologo /std:c11 /experimental:c11atomics /O2 /arch:AVX2 /W3 /Fe:phase0.exe phase0.c cf_model.c cf_json.c
exit /b %errorlevel%
