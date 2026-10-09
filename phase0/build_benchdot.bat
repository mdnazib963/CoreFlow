@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cl /nologo /O2 /arch:AVX2 /W3 /Fe:bench_dot.exe bench_dot.c
if errorlevel 1 exit /b 1
bench_dot.exe 396 4096 64
bench_dot.exe 144 4096 64
