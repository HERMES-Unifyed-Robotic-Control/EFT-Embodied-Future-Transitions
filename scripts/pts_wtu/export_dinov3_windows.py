#!/usr/bin/env python3
"""Offline frozen DINOv3 4×4 features for windows (plan §47.2)."""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    raise NotImplementedError(f"{__file__}: implement against {args.config}")


if __name__ == "__main__":
    main()
