@echo off
rem Run the app with the "chewing" conda env directly, no "conda activate" needed
cd /d "%~dp0"
"%USERPROFILE%\anaconda3\envs\chewing\python.exe" -m streamlit run app.py --server.port 8502
pause
