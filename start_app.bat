@echo off
echo Starting Streamlit app...
streamlit run streamlit_app/app.py --server.port 8501 --server.address localhost
pause 