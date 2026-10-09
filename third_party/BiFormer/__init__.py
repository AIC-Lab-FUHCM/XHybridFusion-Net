from pathlib import Path
import sys


# Upstream imports "models" and "ops" from the repository root.
BIFORMER_ROOT = str(Path(__file__).resolve().parent)

if BIFORMER_ROOT not in sys.path:
    sys.path.insert(0, BIFORMER_ROOT)
