"""Streamlined tensor EDA (see reports/eda-guide.md)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
from sklearn.metrics import r2_score
from tensorly.decomposition import non_negative_parafac, parafac

from cpd_smooth import fit_cpd_smooth, predict_cpd
from fp_data import REPO_ROOT, load_fp_simulation

FIG_DIR = REPO_ROOT / "figures"
RANKS = [2, 3, 4, 6, 8, 10, 12, 16]
PORT_MODE = 5


def tensor_summary(powers: np.ndarray, valid_mask: np.ndarray) -> dict:
    grid_shape = valid_mask.shape
    n_ports = powers.shape[-1]
    n_grid_total = int(valid_mask.size)
    n_grid_valid = int(valid_mask.sum())
    n_scalar_total = n_grid_total * n_ports
    n_scalar_observed = n_grid_valid * n_ports
    return {
        "tensor_shape": tuple(powers.shape),
        "grid_shape": tuple(grid_shape),
        "n_ports": n_ports,
        "n_grid_total": n_grid_total,
        "n_grid_valid": n_grid_valid,
        "n_scalar_total": n_scalar_total,
        "n_scalar_observed": n_scalar_observed,
    }


def print_summary(summary: dict) -> None:
    print("Tensor shape:", summary["tensor_shape"])
    print("Grid cells (valid / total):", summary["n_grid_valid"], "/", summary["n_grid_total"])
    print("Scalar entries (observed / possible):", summary["n_scalar_observed"], "/", summary["n_scalar_total"])


def observed_index_table(powers: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices [N, 6], values [N]) for all valid grid × port entries."""
    grid_idx = np.argwhere(valid_mask)  # (N, 5)
    n = grid_idx.shape[0]
    n_ports = powers.shape[-1]
    rows = []
    vals = []
    for g in range(n):
        i, j, k, l, m = grid_idx[g]
        for p in range(n_ports):
            rows.append([i, j, k, l, m, p])
            vals.append(powers[i, j, k, l, m, p])
    return np.array(rows, dtype=np.int64), np.array(vals, dtype=np.float64)


def masked_relative_error(true: np.ndarray, approx: np.ndarray, mask: np.ndarray) -> float:
    diff = (true - approx)[mask]
    ref = true[mask]
    den = np.linalg.norm(ref)
    return float(np.linalg.norm(diff) / den) if den > 0 else float("nan")


def tensor_from_train_cells(
    powers: np.ndarray, valid_mask: np.ndarray, train_grid: set[tuple[int, ...]]
) -> np.ndarray:
    out = np.zeros_like(powers)
    for idx in train_grid:
        out[idx] = powers[idx]
    return out


def recon_tensorly(tensor: np.ndarray, rank: int, random_state: int = 0) -> np.ndarray:
    cp = parafac(
        tensor,
        rank=rank,
        init="random",
        random_state=random_state,
        n_iter_max=300,
        tol=1e-7,
    )
    return tl.cp_to_tensor(cp)


def recon_tensorly_nn(tensor: np.ndarray, rank: int, random_state: int = 0) -> np.ndarray:
    cp = non_negative_parafac(
        tensor,
        rank=rank,
        init="random",
        random_state=random_state,
        n_iter_max=300,
        tol=1e-7,
    )
    return tl.cp_to_tensor(cp)


def recon_cpd_on_indices(
    model,
    powers_shape: tuple[int, ...],
    valid_mask: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Fill full tensor with model predictions at observed index layout."""
    grid_idx = np.argwhere(valid_mask)
    n_ports = powers_shape[-1]
    out = np.zeros(powers_shape, dtype=np.float64)
    rows = []
    positions = []
    for g in grid_idx:
        i, j, k, l, m = g
        for p in range(n_ports):
            rows.append([i, j, k, l, m, p])
            positions.append((i, j, k, l, m, p))
    if not rows:
        return out
    pred = predict_cpd(model, np.array(rows), device=device)
    for val, pos in zip(pred, positions):
        out[pos] = val
    return out


def plot_flat_histogram(powers: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(powers.ravel(), bins=80, color="steelblue", edgecolor="white")
    ax.set_xlabel("Power (W)")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of powers.ravel() (includes invalid zeros)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_four_method_errors(
    ranks: list[int],
    series: dict[str, list[float]],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {
        "CP (TensorLy ALS)": "steelblue",
        "CPD-S (λ=0)": "darkorange",
        "CP-NN (TensorLy HALS)": "crimson",
        "CPD-S (λ=1)": "seagreen",
    }
    for name, errs in series.items():
        ax.plot(ranks, errs, "o-", label=name, color=colors.get(name, None), linewidth=2, markersize=6)
    ax.set_xlabel("Rank")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(ranks)
    ax.set_ylim(bottom=0 if "error" in ylabel.lower() or "R²" in ylabel else None)
    if "R²" in ylabel:
        ax.set_ylim(bottom=0, top=1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def full_tensor_benchmark(
    powers: np.ndarray,
    valid_mask: np.ndarray,
    value_mask: np.ndarray,
    sizes: tuple[int, ...],
    ranks: list[int],
) -> dict[str, list[float]]:
    tensor = np.where(value_mask, powers, 0.0)
    out: dict[str, list[float]] = {
        "CP (TensorLy ALS)": [],
        "CPD-S (λ=0)": [],
        "CP-NN (TensorLy HALS)": [],
        "CPD-S (λ=1)": [],
    }
    idx_all, y_all = observed_index_table(powers, valid_mask)
    for rank in ranks:
        print(f"  full fit rank {rank}")
        recon_tl = recon_tensorly(tensor, rank)
        out["CP (TensorLy ALS)"].append(masked_relative_error(powers, recon_tl, value_mask))

        recon_nn = recon_tensorly_nn(tensor, rank)
        out["CP-NN (TensorLy HALS)"].append(masked_relative_error(powers, recon_nn, value_mask))

        m0 = fit_cpd_smooth(idx_all, y_all, sizes, rank, smooth_lambda=0.0, n_epochs=150)
        recon_s0 = recon_cpd_on_indices(m0, powers.shape, valid_mask)
        out["CPD-S (λ=0)"].append(masked_relative_error(powers, recon_s0, value_mask))

        m1 = fit_cpd_smooth(idx_all, y_all, sizes, rank, smooth_lambda=1.0, n_epochs=150)
        recon_s1 = recon_cpd_on_indices(m1, powers.shape, valid_mask)
        out["CPD-S (λ=1)"].append(masked_relative_error(powers, recon_s1, value_mask))

    return out


def holdout_r2_benchmark(
    powers: np.ndarray,
    valid_mask: np.ndarray,
    sizes: tuple[int, ...],
    ranks: list[int],
    train_frac: float = 0.8,
    random_state: int = 0,
) -> dict[str, list[float]]:
    grid_idx = [tuple(x) for x in np.argwhere(valid_mask)]
    rng = np.random.default_rng(random_state)
    rng.shuffle(grid_idx)
    n_train = int(len(grid_idx) * train_frac)
    train_set = set(grid_idx[:n_train])
    test_set = set(grid_idx[n_train:])

    train_rows, train_y = [], []
    test_rows, test_y = [], []
    n_ports = powers.shape[-1]
    for cell in train_set:
        i, j, k, l, m = cell
        for p in range(n_ports):
            train_rows.append([i, j, k, l, m, p])
            train_y.append(powers[i, j, k, l, m, p])
    for cell in test_set:
        i, j, k, l, m = cell
        for p in range(n_ports):
            test_rows.append([i, j, k, l, m, p])
            test_y.append(powers[i, j, k, l, m, p])

    train_idx = np.array(train_rows, dtype=np.int64)
    train_y = np.array(train_y, dtype=np.float64)
    test_idx = np.array(test_rows, dtype=np.int64)
    test_y = np.array(test_y, dtype=np.float64)

    tensor_train = tensor_from_train_cells(powers, valid_mask, train_set)
    value_mask = np.broadcast_to(valid_mask[..., None], powers.shape)

    out: dict[str, list[float]] = {
        "CP (TensorLy ALS)": [],
        "CPD-S (λ=0)": [],
        "CP-NN (TensorLy HALS)": [],
        "CPD-S (λ=1)": [],
    }

    for rank in ranks:
        print(f"  holdout rank {rank}")
        recon_tl = recon_tensorly(tensor_train, rank)
        pred_tl = recon_tl[test_idx[:, 0], test_idx[:, 1], test_idx[:, 2], test_idx[:, 3], test_idx[:, 4], test_idx[:, 5]]
        out["CP (TensorLy ALS)"].append(float(r2_score(test_y, pred_tl)))

        recon_nn = recon_tensorly_nn(tensor_train, rank)
        pred_nn = recon_nn[test_idx[:, 0], test_idx[:, 1], test_idx[:, 2], test_idx[:, 3], test_idx[:, 4], test_idx[:, 5]]
        out["CP-NN (TensorLy HALS)"].append(float(r2_score(test_y, pred_nn)))

        m0 = fit_cpd_smooth(train_idx, train_y, sizes, rank, smooth_lambda=0.0, n_epochs=150)
        pred_s0 = predict_cpd(m0, test_idx)
        out["CPD-S (λ=0)"].append(float(r2_score(test_y, pred_s0)))

        m1 = fit_cpd_smooth(train_idx, train_y, sizes, rank, smooth_lambda=1.0, n_epochs=150)
        pred_s1 = predict_cpd(m1, test_idx)
        out["CPD-S (λ=1)"].append(float(r2_score(test_y, pred_s1)))

    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_fp_simulation()
    powers = data["powers"]
    valid_mask = data["valid_mask"]
    value_mask = np.broadcast_to(valid_mask[..., None], powers.shape)
    sizes = tuple(powers.shape)

    summary = tensor_summary(powers, valid_mask)
    print_summary(summary)

    plot_flat_histogram(powers, FIG_DIR / "eda_tensor_hist.png")
    print(f"Wrote {FIG_DIR / 'eda_tensor_hist.png'}")

    print("Full-tensor reconstruction error (masked valid entries)...")
    full_series = full_tensor_benchmark(powers, valid_mask, value_mask, sizes, RANKS)
    plot_four_method_errors(
        RANKS,
        full_series,
        ylabel="Masked relative Frobenius error",
        title="CP reconstruction error vs rank (all observed data)",
        out_path=FIG_DIR / "eda_cp_error_by_rank.png",
    )
    print(f"Wrote {FIG_DIR / 'eda_cp_error_by_rank.png'}")

    print("80/20 grid holdout R²...")
    holdout_series = holdout_r2_benchmark(powers, valid_mask, sizes, RANKS)
    plot_four_method_errors(
        RANKS,
        holdout_series,
        ylabel="R² (holdout 20%)",
        title="Holdout R² vs rank (train on 80% of valid grid cells)",
        out_path=FIG_DIR / "eda_cp_holdout_r2.png",
    )
    print(f"Wrote {FIG_DIR / 'eda_cp_holdout_r2.png'}")

    for name, vals in full_series.items():
        print(name, [f"{v:.4f}" for v in vals])
    for name, vals in holdout_series.items():
        print(name + " R2", [f"{v:.4f}" for v in vals])


if __name__ == "__main__":
    main()
