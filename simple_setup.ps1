# Enable script execution
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Write-Host "Setting up environment..."

# Activate virtual environment
Write-Host "Activating virtual environment..."
& .\fresh_env\Scripts\Activate.ps1

# Install packages
Write-Host "Installing packages..."
pip install -r simple_requirements.txt

Write-Host "Starting Streamlit app..."
streamlit run streamlit_app/app.py --server.port 8501 --server.address localhost

Read-Host -Prompt "Press Enter to exit" 