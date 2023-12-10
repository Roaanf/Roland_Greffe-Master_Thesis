@echo off

rem **************************************************************************
rem USER : Choose your compiler
rem **************************************************************************

rem Call the script selecting the CMAKE compiler (Visual Studio 2010, 2012, etc...)
call cmakeGeneratorSettings.bat

rem **************************************************************************
rem CMAKE GENERATION
rem **************************************************************************

if %GV_COMPILER%==Visual_Studio_12_Win64 (
    call :generate_Visual_Studio_12_Win64
	goto :finish
) else if %GV_COMPILER%==Visual_Studio_16_2019_Win64 (
    call :generate_Visual_Studio_16_2019_Win64
	goto :finish
) else if %GV_COMPILER%==Visual_Studio_17_2022_Win64 (
    call :generate_Visual_Studio_17_2022_Win64
	goto :finish
)

:generate_Visual_Studio_12_Win64
set CURRENTSCRIPTPATH=%CD%
cd ..
cd ..
mkdir Generated_VC12_x64
cd Generated_VC12_x64
mkdir Tutorials
cd Tutorials
mkdir ViewerPlugins
cd ViewerPlugins
rem CMake the application
cmake -G "Visual Studio 12 Win64" ..\..\..\Development\Tutorials\ViewerPlugins
if NOT ERRORLEVEL 0 pause
cd %CURRENTSCRIPTPATH%
pause
goto :finish

:generate_Visual_Studio_16_2019_Win64
set CURRENTSCRIPTPATH=%CD%
cd ..
cd ..
mkdir Generated_VC19_x64
cd Generated_VC19_x64
mkdir Tutorials
cd Tutorials
mkdir ViewerPlugins
cd ViewerPlugins
rem CMake the application
cmake -G "Visual Studio 16 2019" -A x64 ..\..\..\Development\Tutorials\ViewerPlugins
if NOT ERRORLEVEL 0 pause
cd %CURRENTSCRIPTPATH%
pause
goto :finish

:generate_Visual_Studio_17_2022_Win64
set CURRENTSCRIPTPATH=%CD%
cd ..
cd ..
mkdir Generated_VC22_x64
cd Generated_VC22_x64
mkdir Tutorials
cd Tutorials
mkdir ViewerPlugins
cd ViewerPlugins
rem CMake the application
cmake -G "Visual Studio 17 2022" -A x64 ..\..\..\Development\Tutorials\ViewerPlugins
if NOT ERRORLEVEL 0 pause
cd %CURRENTSCRIPTPATH%
pause
goto :finish

:finish
