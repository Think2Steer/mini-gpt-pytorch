import sys

print("Python executable:", sys.executable)

try:
    import torch
except ModuleNotFoundError:
    print("\nERROR: torch is not installed in this interpreter.")
    print("Fix:")
    print("1) Select the project venv interpreter in your IDE.")
    print("2) Or run: python -m pip install -r requirements.txt")
    print("Expected interpreter path should look like:")
    print("<repo-root>/venv/bin/python")
    raise SystemExit(1)

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
