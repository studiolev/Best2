# Enable script execution
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Write-Host "Activating virtual environment..."
& .\streamlit_env\Scripts\Activate.ps1

Write-Host "Installing packages..."
pip install --upgrade pip
pip install wheel
pip install streamlit==1.32.0
pip install numpy==1.26.4
pip install pandas==2.0.3
pip install pillow==10.0.0
pip install protobuf==4.21.12
pip install requests==2.31.0

Write-Host "Starting Streamlit app..."
streamlit run streamlit_app/app.py

Read-Host -Prompt "Press Enter to exit" 