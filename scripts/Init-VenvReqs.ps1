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

Write-Host " - Install requirements"
pip install -r requirements.txt
#########################################################################################
Write-Host "<< $scriptName - Done"
