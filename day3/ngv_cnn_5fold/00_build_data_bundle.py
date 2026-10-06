"""TEACHER ONLY: acquire real sources once, create MNIST 2000 / CIFAR-10 10000 / Oxford cats-dogs 2000.
This is NOT an already data-included package. ALLOW_DOWNLOAD=True explicitly
allows source download on this computer. Set False when using local originals.
After success, run 08_make_student_zip.py; students then skip this script.
"""
from cnnlab.builder import build_all

ALLOW_DOWNLOAD = True
SOURCE_DIR = None  # e.g. r"C:\Users\your_name\Downloads\dataset_sources"

if __name__ == "__main__":
    build_all(source_dir=SOURCE_DIR, allow_download=ALLOW_DOWNLOAD)
