@echo off
:: 如果 PYTHON 环境变量未定义则 设置该变量为 python
if not defined PYTHON (set PYTHON=python)
:: 如果 VENV_DIR 环境变量未定义则设置为 根目录下的venv目录
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")

set SD_WEBUI_RESTART=tmp/restart
set ERROR_REPORTING=FALSE

:: 在根目录创建tmp文件夹，2>NUL用于文件夹已存时屏蔽报错信息
mkdir tmp 2>NUL

:: python -c "" 表示用python执行语句，验证python环境是否正确
:: >tmp/stdout.txt 表示执行成功的话，将结果输出到tmp/stdout.txt文件
:: >tmp/stderr.txt 表示执行错误的话，将结果输出到tmp/stderr.txt文件
%PYTHON% -c "" >tmp/stdout.txt 2>tmp/stderr.txt
:: 上一条语句执行成功 %ERRORLEVEL% == 0 位true
:: 上一条语句执行成功，跳转 check_pip 标签
if %ERRORLEVEL% == 0 goto :check_pip
echo Couldn't launch python
:: python -c "" 执行出现err则跳转 show_stdout_stderr 标签
goto :show_stdout_stderr

:check_pip
:: 执行python -mpip --help 命令
%PYTHON% -mpip --help >tmp/stdout.txt 2>tmp/stderr.txt
:: 执行pip命令没出错，则跳转 start_venv 标签
if %ERRORLEVEL% == 0 goto :start_venv
:: pip安装路径为空串，则跳转 show_stdout_stderr
if "%PIP_INSTALLER_LOCATION%" == "" goto :show_stdout_stderr
%PYTHON% "%PIP_INSTALLER_LOCATION%" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :start_venv
echo Couldn't install pip
goto :show_stdout_stderr

:start_venv
:: 启动虚拟环境
if ["%VENV_DIR%"] == ["-"] goto :skip_venv
if ["%SKIP_VENV%"] == ["1"] goto :skip_venv

:: 执行venv目录下的python命令
dir "%VENV_DIR%\Scripts\Python.exe" >tmp/stdout.txt 2>tmp/stderr.txt
:: 没有出现错误，则跳转 activate_venv 标签
if %ERRORLEVEL% == 0 goto :activate_venv

:: 没有venv虚拟环境则进行创建
for /f "delims=" %%i in ('CALL %PYTHON% -c "import sys; print(sys.executable)"') do set PYTHON_FULLNAME="%%i"
echo Creating venv in directory %VENV_DIR% using python %PYTHON_FULLNAME%
%PYTHON_FULLNAME% -m venv "%VENV_DIR%" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :activate_venv
echo Unable to create venv in directory "%VENV_DIR%"
goto :show_stdout_stderr

:activate_venv
:: 激活虚拟环境
set PYTHON="%VENV_DIR%\Scripts\Python.exe"
echo venv %PYTHON%

:skip_venv
if [%ACCELERATE%] == ["True"] goto :accelerate
goto :launch

:accelerate
echo Checking for accelerate
set ACCELERATE="%VENV_DIR%\Scripts\accelerate.exe"
if EXIST %ACCELERATE% goto :accelerate_launch

:launch
:: python launch.py 参数
%PYTHON% launch.py %*
if EXIST tmp/restart goto :skip_venv
pause
exit /b

:accelerate_launch
echo Accelerating
%ACCELERATE% launch --num_cpu_threads_per_process=6 launch.py
if EXIST tmp/restart goto :skip_venv
pause
exit /b

:show_stdout_stderr

echo.
echo exit code: %errorlevel%

for /f %%i in ("tmp\stdout.txt") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stdout:
type tmp\stdout.txt

:show_stderr
for /f %%i in ("tmp\stderr.txt") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stderr:
type tmp\stderr.txt

:endofscript

echo.
echo Launch unsuccessful. Exiting.
pause
