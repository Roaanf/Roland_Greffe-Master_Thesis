@echo on

rem **************************************************************************
rem USER : Choose your compiler
rem **************************************************************************

rem Call the script selecting the CMAKE compiler (Visual Studio 2010, 2012, etc...)
call cmakeGeneratorSettings.bat

rem **************************************************************************
rem Unzip Externals
rem **************************************************************************

set GS_EXTERNALS_FILE=windows-VS2013-x64.7z

set CURRENTSCRIPTPATH=%CD%
cd ..
cd ..
cd External
cd Windows
rem 7-Zip settings
rem e : extract files from archive
rem -y : assume Yes on all queries
rem call "%ProgramFiles(x86)%\7-zip\7z.exe" x -y %GS_EXTERNALS_FILE%
call "%GV_7ZIP_PATH%\7z.exe" x -y %GS_EXTERNALS_FILE%

cd %CURRENTSCRIPTPATH%

pause
