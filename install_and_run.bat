@echo off
echo Installing dependencies...
call streamlit_env\Scripts\activate.bat
pip install -r streamlit_requirements.txt
echo Starting Streamlit app...
streamlit run streamlit_app/app.py --server.port 8501
pause 