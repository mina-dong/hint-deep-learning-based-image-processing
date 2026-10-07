"""CIFAR-10: four randomly initialized CNN variants
VS Code: select the project Python interpreter, open this file, Run Python File.
Edit hyperparameters in configs/exp03_models.json. No terminal arguments are required.
"""
from cnnlab.runner import run_plan

DEVICE = "auto"       # "auto", "cpu", or "cuda". Use "cpu" if GPU setup fails.
OPEN_REPORT = True    # Open the completed local HTML report in your browser.
CONFIG_FILE = "exp03_models.json"

if __name__ == "__main__":
    run_plan(CONFIG_FILE, device=DEVICE, open_report=OPEN_REPORT)

#gpu가 아닌 cpu로 처리하려하다보니 오류 발생으로 전체 실습코드 업로드