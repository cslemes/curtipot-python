"""Streamlit Community Cloud entrypoint for CurTiPot Python."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from curtipot.app import main

if __name__ == "__main__":
    main()
