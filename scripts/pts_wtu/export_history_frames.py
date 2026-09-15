#!/usr/bin/env python3
"""Export history RGB + future visual anchors (plan §47.1–2)."""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    raise NotImplementedError(f"{__file__}: implement against {args.config}")


if __name__ == "__main__":
    main()
