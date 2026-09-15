#!/usr/bin/env python3
"""Frozen-U linear probes + dump metrics (plan §36, §47.9)."""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    raise NotImplementedError(f"{__file__}: implement against {args.config}")


if __name__ == "__main__":
    main()
