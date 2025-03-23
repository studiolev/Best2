# Enable script execution if needed
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Write-Host "Activating virtual environment..."
& .\streamlit_env\Scripts\Activate.ps1

Write-Host "Installing dependencies..."
pip install -r streamlit_requirements.txt

Write-Host "Starting Streamlit app..."
streamlit run streamlit_app/app.py --server.port 8501

Read-Host -Prompt "Press Enter to exit" 