#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evomax_runner import resolve_stage1_limit


def main() -> int:
    fractional = {"top_fraction_mid": 0.015, "top_k_mid": None}
    assert resolve_stage1_limit(fractional, 9_405) == 141
    assert resolve_stage1_limit(fractional, 57) == 1

    fixed_override = {"top_fraction_mid": 0.015, "top_k_mid": 100}
    assert resolve_stage1_limit(fixed_override, 9_405) == 100
    assert resolve_stage1_limit(fixed_override, 57) == 57

    print("Stage 1 selection tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
