# Stop any existing Python processes
Get-Process | Where-Object {$_.ProcessName -eq "python"} | Stop-Process -Force

# Set execution policy for current user
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force

# Create and activate new virtual environment
Write-Host "Creating new virtual environment..." -ForegroundColor Green
python -m venv streamlit_env
$env:PATH = "$pwd\streamlit_env\Scripts;$env:PATH"
.\streamlit_env\Scripts\activate.ps1

# Install dependencies using wheels
Write-Host "Installing dependencies..." -ForegroundColor Green
python -m pip install --upgrade pip wheel setuptools
pip install --only-binary :all: numpy==1.24.3
pip install --only-binary :all: pandas==2.0.3
pip install streamlit==1.32.0 pillow==10.0.0 protobuf==4.21.12 requests==2.31.0

# Start Streamlit in a new window
Write-Host "Starting Streamlit..." -ForegroundColor Green
$streamlitCmd = "cd '$pwd\streamlit_app'; & '$pwd\streamlit_env\Scripts\activate.ps1'; python -m streamlit run app.py"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $streamlitCmd

# Start FastAPI in a new window
Write-Host "Starting FastAPI..." -ForegroundColor Green
$fastApiCmd = "cd '$pwd'; & '$pwd\venv_new\Scripts\activate.ps1'; uvicorn app.main:app --reload"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $fastApiCmd

Write-Host "Setup complete! The applications should be running at:" -ForegroundColor Green
Write-Host "Streamlit UI: http://localhost:8501" -ForegroundColor Cyan
Write-Host "FastAPI Backend: http://localhost:8000" -ForegroundColor Cyan 