from pathlib import Path
import sys


# Source gốc dùng import "models" và "ops" từ thư mục repository.
BIFORMER_ROOT = str(Path(__file__).resolve().parent)

if BIFORMER_ROOT not in sys.path:
    sys.path.insert(0, BIFORMER_ROOT)
