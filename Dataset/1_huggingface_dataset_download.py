import subprocess
import sys
import os

REPO_URL = "https://huggingface.co/datasets/jrobe187/cv_final_project_group1"
TARGET_DIR = "cv_final_project_group1"

def run(cmd):
    """Run a shell command and stream its output."""
    print(f"Running: {cmd}")
    process = subprocess.Popen(cmd, shell=True)
    process.communicate()

# 1. Install git-lfs if needed
print("Checking git-lfs installation...")
result = subprocess.run("git lfs --version", shell=True)

if result.returncode != 0:
    print("git-lfs not found. Installing...")
    # Linux/Colab install
    run("sudo apt-get update && sudo apt-get install -y git-lfs")
    run("git lfs install")
else:
    print("git-lfs is already installed.")

# 2. Clone the dataset
if not os.path.exists(TARGET_DIR):
    print(f"Cloning repository to {TARGET_DIR}...")
    run(f"git clone {REPO_URL}")
else:
    print(f"Directory '{TARGET_DIR}' already exists. Skipping clone.")

# 3. Pull LFS files
print("Pulling LFS files (this downloads the dataset)...")
run(f"cd {TARGET_DIR} && git lfs pull")

print("Download complete!")
