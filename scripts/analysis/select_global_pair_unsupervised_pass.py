from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis import select_global_pair_unsupervised as base


def main() -> None:
    argv = sys.argv[1:]
    if not any(arg == "--selection_rules" or arg.startswith("--selection_rules=") for arg in argv):
        argv = [*argv, "--selection_rules", "pass_metric"]
    sys.argv = [sys.argv[0], *argv]
    base.main()


if __name__ == "__main__":
    main()
