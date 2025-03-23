import subprocess
import sys
import os

def run_command(command):
    print(f"Running: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def main():
    # Create virtual environment if it doesn't exist
    if not os.path.exists("venv"):
        print("Creating virtual environment...")
        run_command(f"{sys.executable} -m venv venv")

    # Activate virtual environment and install packages
    if sys.platform == "win32":
        activate_cmd = ".\\venv\\Scripts\\activate"
    else:
        activate_cmd = "source ./venv/bin/activate"

    print("Installing packages...")
    run_command(f"{activate_cmd} && pip install streamlit==1.32.0 pillow==10.0.0 requests==2.31.0")

    print("Starting Streamlit app...")
    run_command(f"{activate_cmd} && streamlit run streamlit_app/app.py --server.port 8501 --server.address localhost")

if __name__ == "__main__":
    main() 