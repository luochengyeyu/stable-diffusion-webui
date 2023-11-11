@echo off

set PYTHON=
set GIT=
set VENV_DIR=
:: set COMMANDLINE_ARGS=--loglevel DEBUG --log-startup --listen --xformers --enable-console-prompts
set COMMANDLINE_ARGS=--listen --xformers --enable-console-prompts

call webui.bat
