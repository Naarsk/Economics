$scriptName = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)
Write-Host ">> $scriptName - Start"
$MyDir = [System.IO.Path]::GetDirectoryName($myInvocation.MyCommand.Definition)
Set-Location $MyDir\..\
#########################################################################################

Write-Host " - Init Venv"
python -m venv venv

Write-Host " - Activate Venv"
venv\Scripts\Activate

Write-Host " - Upgrade pip"
python.exe -m pip install --upgrade pip

Write-Host " - Install pydantic pandas openpyxl QuantLib"
pip install pydantic pandas openpyxl QuantLib

Write-Host " - Install Tests and code style"
pip install pycodestyle coverage
# pip install pylint

Write-Host " - Install build twine"
pip install build twine

#########################################################################################
Write-Host "<< $scriptName - Done"
