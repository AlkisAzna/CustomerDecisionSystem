@echo off

echo Creating python environments and downloading necessary packages...
echo.
echo Enter password when prompted and press enter
echo Don't worry if nothing appears on the screen while typing. That's the way it works ;)
echo.

python -m venv %USERPROFILE%\CustomerSystem
call %USERPROFILE%\CustomerSystem\Scripts\activate.bat
%USERPROFILE%\CustomerSystem\Scripts\python.exe -m pip install --upgrade pip
pip install pandas=1.1.5 openpyxl=3.0.5 xlrd=1.2.0 matplotlib
call %USERPROFILE%\CustomerSystem\Scripts\deactivate.bat

echo.
pause