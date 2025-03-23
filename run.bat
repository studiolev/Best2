@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing packages...
pip install --upgrade pip wheel setuptools
pip install --only-binary :all: streamlit==1.32.0
pip install --only-binary :all: pillow==10.0.0
pip install --only-binary :all: requests==2.31.0

echo Starting Streamlit app...
streamlit run streamlit_app/app.py --server.port 8501 --server.address localhost

pause 
