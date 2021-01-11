@echo off

echo Creating python environments and downloading necessary packages...
echo.
echo Enter password when prompted and press enter
echo Don't worry if nothing appears on the screen while typing. That's the way it works ;)
echo.

python -m venv %USERPROFILE%\CustomerSystem
cd %USERPROFILE%\CustomerSystem
git clone https://github.com/AlkisAzna/CustomerDecisionSystem.git

call %USERPROFILE%\CustomerSystem\Scripts\activate.bat
%USERPROFILE%\CustomerSystem\Scripts\python.exe -m pip install --upgrade pip
pip install -r %USERPROFILE%\CustomerSystem\CustomerDecisionSystem\requirements.txt
call %USERPROFILE%\CustomerSystem\Scripts\deactivate.bat

echo.
pause


@echo off

%USERPROFILE%\CustomerSystem\Scripts\python.exe %USERPROFILE%\CustomerSystem\CustomerDecisionSystem\main_system.py

pause
