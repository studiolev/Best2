@echo off
echo Creating conda environment...
conda create -n furniture_app python=3.10 -y

echo Activating conda environment...
call conda activate furniture_app

echo Installing packages...
conda install -c conda-forge streamlit pillow requests -y

echo Starting Streamlit app...
streamlit run streamlit_app/app.py --server.port 8501 --server.address localhost

pause 