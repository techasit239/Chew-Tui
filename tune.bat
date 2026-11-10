@echo off
rem Find the best counting settings from recorded trials (data\raw) using the "chewing" conda env
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
"%USERPROFILE%\anaconda3\envs\chewing\python.exe" tune.py %*
pause
