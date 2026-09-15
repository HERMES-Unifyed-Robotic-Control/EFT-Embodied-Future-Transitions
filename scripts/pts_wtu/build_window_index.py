#!/usr/bin/env python3
"""Build (task, ep, t) index for predictive windows (plan §47.1)."""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    raise NotImplementedError(f"{__file__}: implement against {args.config}")


if __name__ == "__main__":
    main()
