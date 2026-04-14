@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help
if "%1" == "help" goto help
if "%1" == "clean" goto clean
if "%1" == "dirhtml" goto dirhtml
if "%1" == "html" goto html
if "%1" == "htmlhelp" goto htmlhelp
if "%1" == "qthelp" goto qthelp
if "%1" == "devhelp" goto devhelp
if "%1" == "epub" goto epub
if "%1" == "latex" goto latex
if "%1" == "latexpdf" goto latexpdf
if "%1" == "text" goto text
if "%1" == "man" goto man
if "%1" == "texinfo" goto texinfo
if "%1" == "gettext" goto gettext
if "%1" == "moxedoc" goto moxedoc

%SPHINXBUILD% >nul 2>nul
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the sphinx-build executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "clean" (
    del /q /s "%BUILDDIR%\*"
    goto end
)

if "%1" == "html" (
    %SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
    goto end
)

if "%1" == "dirhtml" (
    %SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
    goto end
)

if "%1" == "htmlhelp" (
    %SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
    goto end
)

if "%1" == "qthelp" (
    %SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
    goto end
)

if "%1" == "devhelp" (
    %SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
    goto end
)

if "%1" == "epub" (
    %SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
    goto end
)

if "%1" == "latex" (
    %SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
    goto end
)

if "%1" == "latexpdf" (
    %SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
    goto end
)

if "%1" == "text" (
    %SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
    goto end
)

if "%1" == "man" (
    %SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
    goto end
)

if "%1" == "texinfo" (
    %SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
    goto end
)

if "%1" == "gettext" (
    %SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
    goto end
)

if "%1" == "moxedoc" (
    %SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
    goto end
)

:end
popd