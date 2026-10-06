"""MNIST / LeNet-inspired small CNN
VS Code: select the project Python interpreter, open this file, Run Python File.
Edit hyperparameters in configs/exp01_mnist.json. No terminal arguments are required.
"""
from cnnlab.runner import run_plan

DEVICE = "auto"       # "auto", "cpu", or "cuda". Use "cpu" if GPU setup fails.
OPEN_REPORT = True    # Open the completed local HTML report in your browser.
CONFIG_FILE = "exp01_mnist.json"

if __name__ == "__main__":
    run_plan(CONFIG_FILE, device=DEVICE, open_report=OPEN_REPORT)
