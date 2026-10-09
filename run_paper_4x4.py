"""Generate the full artifact set using paper-derived 4x4 inputs."""

from __future__ import annotations

import sys

from quantum_paper_artifacts import main


if __name__ == "__main__":
    raise SystemExit(main(["--mode", "paper", *sys.argv[1:]]))
