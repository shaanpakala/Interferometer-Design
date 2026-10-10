"""Run all Fabry–Perot tensor EDA scripts."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import fp_eda_decomposition
import fp_eda_histograms
import fp_eda_overview


def main() -> None:
    fp_eda_histograms.main()
    fp_eda_overview.main()
    fp_eda_decomposition.main()


if __name__ == "__main__":
    main()
