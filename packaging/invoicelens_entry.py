"""Entry point for the standalone build. Kept separate from __main__.py because
PyInstaller needs a plain script, not a package module."""

import sys

from invoicelens.cli import main

if __name__ == "__main__":
    sys.exit(main())
