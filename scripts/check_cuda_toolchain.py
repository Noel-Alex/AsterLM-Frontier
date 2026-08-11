#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from asterlm.cuda_toolchain import cuda_toolchain_report, require_compatible_cuda_toolchain


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the pip CUDA extension-build toolchain")
    parser.add_argument("--require-compatible", action="store_true")
    args = parser.parse_args()
    report = (
        require_compatible_cuda_toolchain() if args.require_compatible else cuda_toolchain_report()
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
