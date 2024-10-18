$scriptName = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)
Write-Host ">> $scriptName - Start"
$MyDir = [System.IO.Path]::GetDirectoryName($myInvocation.MyCommand.Definition)
Set-Location $MyDir\..\
#########################################################################################

Write-Host " - Init Venv"
python -m venv venv

Write-Host " - Activate Venv"
venv\Scripts\Activate

Write-Host " - Install notebook, matplotlib and jupyter"
pip install notebook matplotlib jupyter

#########################################################################################
Write-Host "<< $scriptName - Done"
