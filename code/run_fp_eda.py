"""Run all Fabry–Perot tensor EDA scripts."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import fp_eda_core


def main() -> None:
    fp_eda_core.main()


if __name__ == "__main__":
    main()
