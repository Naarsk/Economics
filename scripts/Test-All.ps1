$scriptName = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)
Write-Host ">> $scriptName - Start"
$MyDir = [System.IO.Path]::GetDirectoryName($myInvocation.MyCommand.Definition)
Set-Location $MyDir\..\
#########################################################################################

Write-Host " - Activate Venv"
venv\Scripts\Activate

Write-Host " - coverage run -m unittest"
python -m coverage run -m unittest

Write-Host " - coverage report --show-missing"
python -m coverage report --show-missing

Write-Host " - coverage html"
python -m coverage html

Write-Host " - pycodestyle pythflib --max-line-length=120"
python -m pycodestyle pythflib --max-line-length=120

Write-Host " - pycodestyle tests --max-line-length=120"
python -m pycodestyle tests --max-line-length=120

# disabled for now otherwise I have to fix too many things :)
# Write-Host " - pylint pythflib"
# python -m pylint pythflib

#########################################################################################
Write-Host "<< $scriptName - Done"

