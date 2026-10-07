"""TEACHER ONLY: create a WITH_DATA ZIP only after all 14000 real samples pass checks."""
from cnnlab.builder import make_student_zip
if __name__ == "__main__":
    make_student_zip()
