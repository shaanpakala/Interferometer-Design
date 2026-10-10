"""Histogram EDA for Fabry–Perot power tensor."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fp_data import AXIS_NAMES, REPO_ROOT, load_fp_simulation

FIG_DIR = REPO_ROOT / "figures"


def _valid_powers(powers: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Flatten port powers over valid grid cells only."""
    # powers: (*grid, n_ports)
    n_ports = powers.shape[-1]
    flat = powers.reshape(-1, n_ports)
    mask = valid_mask.ravel()
    return flat[mask]


def plot_marginal_histograms(data: dict, out_dir: Path) -> None:
    vals = _valid_powers(data["powers"], data["valid_mask"]).ravel()
    vals_pos = vals[vals > 0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(vals, bins=50, color="steelblue", edgecolor="white", alpha=0.9)
    axes[0].set_xlabel("Power (W)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("All valid port powers")

    if vals_pos.size:
        axes[1].hist(np.log10(vals_pos), bins=50, color="darkorange", edgecolor="white", alpha=0.9)
        axes[1].set_xlabel(r"$\log_{10}$ power (W)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Positive powers (log scale)")
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "hist_marginal_powers.png", dpi=150)
    plt.close(fig)


def plot_disaggregated_by_mode(data: dict, out_dir: Path) -> None:
    powers = data["powers"]
    valid_mask = data["valid_mask"]
    axes = data["axes"]

    # One figure per parameter mode: overlay histograms for each discrete index along that axis.
    for mode_idx, axis_name in enumerate(AXIS_NAMES):
        n_levels = powers.shape[mode_idx]
        fig, ax = plt.subplots(figsize=(7, 4))
        cmap = plt.cm.viridis(np.linspace(0.15, 0.9, n_levels))

        for level in range(n_levels):
            sl_grid = [slice(None)] * valid_mask.ndim
            sl_grid[mode_idx] = level
            vm = valid_mask[tuple(sl_grid)].ravel()
            block = powers[tuple(sl_grid + [slice(None)])].reshape(-1, powers.shape[-1])
            block = block[vm].ravel()
            block = block[block > 0]
            if block.size == 0:
                continue
            label = f"{axis_name}={axes[axis_name][level]:.3g}"
            ax.hist(
                np.log10(block),
                bins=30,
                histtype="step",
                linewidth=1.4,
                color=cmap[level],
                label=label,
                density=True,
            )

        ax.set_xlabel(r"$\log_{10}$ power (W)")
        ax.set_ylabel("Density")
        ax.set_title(f"Power distribution by {axis_name} level (valid cells)")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / f"hist_by_{axis_name}.png", dpi=150)
        plt.close(fig)

    # By port (last mode)
    port_names = data["port_names"]
    vals = _valid_powers(powers, valid_mask)
    fig, ax = plt.subplots(figsize=(8, 4))
    for p, name in enumerate(port_names):
        col = vals[:, p]
        col = col[col > 0]
        if col.size:
            ax.hist(
                np.log10(col),
                bins=25,
                histtype="step",
                linewidth=1.2,
                label=name,
                density=True,
            )
    ax.set_xlabel(r"$\log_{10}$ power (W)")
    ax.set_ylabel("Density")
    ax.set_title("By port (valid cells)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "hist_by_port.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_fp_simulation()
    plot_marginal_histograms(data, FIG_DIR)
    plot_disaggregated_by_mode(data, FIG_DIR)
    print(f"Wrote histogram figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
