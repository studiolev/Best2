# Enable script execution
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Write-Host "Setting up fresh environment..."

# Activate virtual environment
Write-Host "Activating virtual environment..."
& .\fresh_env\Scripts\Activate.ps1

# Upgrade pip and install wheel
Write-Host "Upgrading pip and installing wheel..."
python -m pip install --upgrade pip
pip install wheel

# Install packages one by one with specific versions
Write-Host "Installing packages..."
pip install streamlit==1.32.0
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install pillow==10.0.0
pip install protobuf==4.21.12
pip install requests==2.31.0

# Create necessary directories if they don't exist
Write-Host "Creating necessary directories..."
if (-not (Test-Path "streamlit_app")) {
    New-Item -ItemType Directory -Path "streamlit_app"
}

# Copy app files if they don't exist
Write-Host "Setting up application files..."
if (-not (Test-Path "streamlit_app/app.py")) {
    Copy-Item "app.py" -Destination "streamlit_app/app.py" -Force
}

Write-Host "Starting Streamlit app..."
streamlit run streamlit_app/app.py --server.port 8501 --server.address localhost

Read-Host -Prompt "Press Enter to exit" 