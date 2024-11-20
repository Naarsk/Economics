$scriptName = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)
Write-Host ">> $scriptName - Start"
$MyDir = [System.IO.Path]::GetDirectoryName($myInvocation.MyCommand.Definition)
Set-Location $MyDir\..\

Write-Host " - Create Data Directory"
New-Item -ItemType Directory -Path "$MyDir\data"

Write-Host " - Pull Docker Image"
docker pull rootproject/root:latest

Write-Host " - Run Docker Image"
docker run -it --name Economics_Root -v "$MyDir\data:/root_data" rootproject/root:latest

Write-Host " - Activate Venv"
venv\Scripts\Activate

Write-Host " - Install pyroot"
pip install pyroot