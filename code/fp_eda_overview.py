"""Additional tensor overview plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fp_data import AXIS_NAMES, REPO_ROOT, load_fp_simulation

FIG_DIR = REPO_ROOT / "figures"


def plot_validity_overview(data: dict, out_dir: Path) -> None:
    valid_mask = data["valid_mask"]
    axes = data["axes"]

    # Fraction valid vs LX (marginal)
    n_lx = valid_mask.shape[0]
    frac_lx = [valid_mask[i].mean() for i in range(n_lx)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(n_lx), frac_lx, color="seagreen", edgecolor="white")
    ax.set_xticks(range(n_lx))
    ax.set_xticklabels([f"{v:.2g}" for v in axes["LX"]], rotation=45, ha="right")
    ax.set_xlabel("LX grid value")
    ax.set_ylabel("Fraction of sub-grid stable")
    ax.set_ylim(0, 1.05)
    ax.set_title("Stability rate vs cavity length (other axes full grid)")
    fig.tight_layout()
    fig.savefig(out_dir / "valid_fraction_by_LX.png", dpi=150)
    plt.close(fig)

    # 2D slice: valid at nominal-ish indices on m2_R vs m2_Rc, sum over other dims
    # Fix LX, m1_R, m1_Rc at middle index
    i0 = valid_mask.shape[0] // 2
    i1 = valid_mask.shape[1] // 2
    i2 = valid_mask.shape[2] // 2
    slice_2d = valid_mask[i0, i1, i2, :, :].astype(float)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(slice_2d, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(axes["m2_Rc"])))
    ax.set_xticklabels([f"{v:.2g}" for v in axes["m2_Rc"]], rotation=45, ha="right")
    ax.set_yticks(range(len(axes["m2_R"])))
    ax.set_yticklabels([f"{v:.2g}" for v in axes["m2_R"]])
    ax.set_xlabel("m2_Rc")
    ax.set_ylabel("m2_R")
    ax.set_title(
        f"valid_mask at LX={axes['LX'][i0]:.2g}, "
        f"m1_R={axes['m1_R'][i1]:.2g}, m1_Rc={axes['m1_Rc'][i2]:.2g}"
    )
    fig.colorbar(im, ax=ax, label="valid")
    fig.tight_layout()
    fig.savefig(out_dir / "valid_slice_m2_R_vs_m2_Rc.png", dpi=150)
    plt.close(fig)


def plot_port_summary(data: dict, out_dir: Path) -> None:
    powers = data["powers"]
    valid_mask = data["valid_mask"]
    port_names = data["port_names"]
    n_ports = powers.shape[-1]

    flat = powers.reshape(-1, n_ports)
    mask = valid_mask.ravel()
    vals = flat[mask]

    means = vals.mean(axis=0)
    stds = vals.std(axis=0)

    x = np.arange(n_ports)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, means, yerr=stds, capsize=3, color="slateblue", alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(port_names, rotation=35, ha="right")
    ax.set_ylabel("Power (W)")
    ax.set_title("Mean ± std port power (valid grid cells)")
    fig.tight_layout()
    fig.savefig(out_dir / "port_mean_std.png", dpi=150)
    plt.close(fig)


def plot_axis_sensitivity(data: dict, out_dir: Path) -> None:
    """Mean total circulating power (sum over ports) vs each axis level."""
    powers = data["powers"]
    valid_mask = data["valid_mask"]
    axes = data["axes"]
    total_p = powers.sum(axis=-1)

    fig, axs = plt.subplots(2, 3, figsize=(11, 6))
    axs = axs.ravel()
    for mode_idx, name in enumerate(AXIS_NAMES):
        ax = axs[mode_idx]
        levels = []
        means = []
        for level in range(total_p.shape[mode_idx]):
            sl = [slice(None)] * total_p.ndim
            sl[mode_idx] = level
            block = total_p[tuple(sl)]
            vm = valid_mask[tuple(sl)]
            if vm.any():
                levels.append(axes[name][level])
                means.append(block[vm].mean())
        ax.plot(levels, means, "o-", color="teal")
        ax.set_xlabel(name)
        ax.set_ylabel("Mean Σ ports (W)")
        ax.set_title(f"Sensitivity to {name}")
        ax.grid(True, alpha=0.3)
    axs[-1].set_visible(False)
    fig.suptitle("Marginal mean total power vs parameter (valid cells)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "axis_sensitivity_total_power.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(data: dict, out_dir: Path) -> None:
    """Correlation between parameters and port powers on valid samples."""
    powers = data["powers"]
    valid_mask = data["valid_mask"]
    axes = data["axes"]
    port_names = data["port_names"]

    grid_shape = valid_mask.shape
    n_ports = powers.shape[-1]
    idx = np.argwhere(valid_mask)
    n = idx.shape[0]
    if n < 3:
        return

    param_mat = np.column_stack([axes[name][idx[:, i]] for i, name in enumerate(AXIS_NAMES)])
    power_mat = np.array([powers[tuple(row)][:] for row in idx])

    # params + powers feature matrix
    feats = np.hstack([param_mat, power_mat])
    labels = list(AXIS_NAMES) + port_names
    corr = np.corrcoef(feats, rowvar=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Correlation (valid samples): parameters & port powers")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "corr_params_ports.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_fp_simulation()
    plot_validity_overview(data, FIG_DIR)
    plot_port_summary(data, FIG_DIR)
    plot_axis_sensitivity(data, FIG_DIR)
    plot_correlation_heatmap(data, FIG_DIR)
    print(f"Wrote overview figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
