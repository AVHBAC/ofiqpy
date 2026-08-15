"""Compatibility entry point for the strict live-OFIQ conformance gate."""

from ofiqpy.conformance import main

if __name__ == "__main__":
    raise SystemExit(main())
