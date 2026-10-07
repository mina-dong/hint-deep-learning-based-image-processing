"""Collect only completed local runs into an HTML report and an Excel-readable CSV."""
from cnnlab.runner import collect
if __name__ == "__main__":
    print(collect(open_report=True))
