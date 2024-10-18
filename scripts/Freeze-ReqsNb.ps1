$scriptName = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)
Write-Host ">> $scriptName - Start"
$MyDir = [System.IO.Path]::GetDirectoryName($myInvocation.MyCommand.Definition)
Set-Location $MyDir\..\
#########################################################################################

Write-Host " - Activate Venv"
venv\Scripts\Activate

Write-Host " - Install requirements"
pip freeze > requirements-nb.txt

#########################################################################################
Write-Host "<< $scriptName - Done"
