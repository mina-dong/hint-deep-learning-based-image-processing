from __future__ import annotations
import contextlib, csv, hashlib, json, os, platform, random, time, uuid
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def set_seed(seed: int, threads: int = 4) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Reproducibility is best-effort, not a promise of bitwise CPU/GPU equality.


def resolve_device(requested: str):
    import torch
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("DEVICE must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run 00_check_environment.py, or set DEVICE='cpu'.")
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else
                          "cpu" if requested == "auto" else requested)
    try:
        m = torch.nn.Conv2d(1, 2, 3).to(device)
        m(torch.ones(2, 1, 8, 8, device=device)).sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        del m
    except Exception as exc:
        raise RuntimeError("Actual device computation failed. Set DEVICE='cpu' or check the GPU installation.") from exc
    return device


def environment(device) -> dict:
    import torch, torchvision, sklearn, PIL, matplotlib
    import sys
    return {"python": sys.version.split()[0], "python_executable": sys.executable,
            "platform": platform.platform(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            "sklearn": sklearn.__version__, "pillow": PIL.__version__,
            "matplotlib": matplotlib.__version__, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "cuda_runtime": torch.version.cuda}


@contextlib.contextmanager
def project_lock(root: Path):
    """Block concurrent jobs sharing the project, including teacher data preparation."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".running.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Another job is running, or was forcibly terminated. Lock: {path}. "
                           "Stop all project Python jobs before removing a stale lock.") from None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "created": time.time()}, f)
        yield
    finally:
        path.unlink(missing_ok=True)
