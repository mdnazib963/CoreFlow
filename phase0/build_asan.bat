@echo off
cd /d %~dp0
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 exit /b 1
set PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;%PATH%
cl /nologo /std:c11 /experimental:c11atomics /Zi /DEBUG /fsanitize=address /arch:AVX2 /W3 /Fe:stage_rt_asan.exe stage_rt.c cf_model.c cf_json.c
exit /b %errorlevel%
