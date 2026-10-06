"""Run this file directly in VS Code. Does not download or modify packages."""
import json, sys

DEVICE = "auto"  # Change to "cpu" to test only CPU computation.

if __name__ == "__main__":
    print("Python:", sys.executable, flush=True)
    try:
        from cnnlab.common import resolve_device, environment
        device = resolve_device(DEVICE)
        print(json.dumps(environment(device), ensure_ascii=False, indent=2))
        print("[PASS] Real convolution forward and backward succeeded.")
    except Exception:
        print("[FAILED] Check the selected interpreter and the installation output.",flush=True)
        raise
