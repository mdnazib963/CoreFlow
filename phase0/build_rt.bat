@echo off
cd /d %~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 exit /b 1
cl /nologo /std:c11 /experimental:c11atomics /O2 /arch:AVX2 /W3 /Fe:coreflow.exe stage_rt.c cf_model.c cf_json.c
if errorlevel 1 exit /b 1
cl /nologo /std:c11 /experimental:c11atomics /O2 /arch:AVX2 /W3 /Fe:stage_rt.exe stage_rt.c cf_model.c cf_json.c
exit /b %errorlevel%
