$scriptName = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)
Write-Host ">> $scriptName - Start"
$MyDir = [System.IO.Path]::GetDirectoryName($myInvocation.MyCommand.Definition)
Set-Location $MyDir\..\
#########################################################################################

Write-Host " - Remove coverage"
Remove-Item -Recurse -Force htmlcov -ErrorAction SilentlyContinue
Remove-Item -Path ".coverage" -ErrorAction SilentlyContinue

Write-Host " - Remove dist"
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue

Write-Host " - Remove *.egg-info"
Remove-Item -Recurse -Force *.egg-info -ErrorAction SilentlyContinue

Write-Host " - Remove Python Venv"
Remove-Item -Recurse -Force venv -ErrorAction SilentlyContinue

#########################################################################################
Write-Host "<< $scriptName - Done"
