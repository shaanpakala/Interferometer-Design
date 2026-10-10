"""CP / CP-NN (HALS) reconstruction error vs rank."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac, parafac

from fp_data import REPO_ROOT, load_fp_simulation

FIG_DIR = REPO_ROOT / "figures"
MAX_RANK = 16


def masked_relative_error(
    true: np.ndarray,
    approx: np.ndarray,
    mask: np.ndarray,
) -> float:
    diff = (true - approx)[mask]
    ref = true[mask]
    num = np.linalg.norm(diff)
    den = np.linalg.norm(ref)
    return float(num / den) if den > 0 else float("nan")


def cp_reconstruction_errors(
    tensor: np.ndarray,
    mask: np.ndarray,
    ranks: range,
    n_iter_max: int = 300,
    random_state: int = 0,
) -> tuple[list[int], list[float], list[float]]:
    cp_errors: list[float] = []
    nn_errors: list[float] = []
    rank_list = list(ranks)

    for rank in rank_list:
        cp = parafac(
            tensor,
            rank=rank,
            init="random",
            random_state=random_state,
            n_iter_max=n_iter_max,
            tol=1e-7,
        )
        recon_cp = tl.cp_to_tensor(cp)
        cp_errors.append(masked_relative_error(tensor, recon_cp, mask))

        nn = non_negative_parafac(
            tensor,
            rank=rank,
            init="random",
            random_state=random_state,
            n_iter_max=n_iter_max,
            tol=1e-7,
        )
        recon_nn = tl.cp_to_tensor(nn)
        nn_errors.append(masked_relative_error(tensor, recon_nn, mask))

    return rank_list, cp_errors, nn_errors


def plot_decomposition_errors(
    ranks: list[int],
    cp_errors: list[float],
    nn_errors: list[float],
    out_dir: Path,
) -> None:
    fig, ax_cp = plt.subplots(figsize=(8, 4.5))
    ax_nn = ax_cp.twinx()

    line_cp = ax_cp.plot(
        ranks,
        cp_errors,
        "o-",
        color="steelblue",
        linewidth=2,
        markersize=7,
        label="CP (ALS)",
    )
    line_nn = ax_nn.plot(
        ranks,
        nn_errors,
        "s-",
        color="crimson",
        linewidth=2,
        markersize=7,
        label="CP-NN (HALS)",
    )

    ax_cp.set_xlabel("CP rank")
    ax_cp.set_ylabel("Masked relative Frobenius error (CP)", color="steelblue")
    ax_nn.set_ylabel("Masked relative Frobenius error (CP-NN)", color="crimson")
    ax_cp.tick_params(axis="y", labelcolor="steelblue")
    ax_nn.tick_params(axis="y", labelcolor="crimson")
    ax_cp.set_title("Tensor reconstruction error vs rank (valid cells only)")
    ax_cp.set_xticks(ranks)
    ax_cp.set_ylim(bottom=0)
    ax_nn.set_ylim(bottom=0)
    ax_cp.grid(True, alpha=0.3)

    lines = line_cp + line_nn
    labels = [ln.get_label() for ln in lines]
    ax_cp.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "cp_reconstruction_error.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_fp_simulation()
    powers = data["powers"]
    valid_mask = data["valid_mask"]

    # Invalid cells set to zero; errors computed only where valid_mask is True.
    tensor = np.where(valid_mask[..., None], powers, 0.0)
    mask = np.broadcast_to(valid_mask[..., None], powers.shape)

    ranks = range(1, MAX_RANK + 1)
    rank_list, cp_err, nn_err = cp_reconstruction_errors(tensor, mask, ranks)

    plot_decomposition_errors(rank_list, cp_err, nn_err, FIG_DIR)
    print("Ranks:", rank_list)
    print("CP errors:", [f"{e:.4f}" for e in cp_err])
    print("CP-NN errors:", [f"{e:.4f}" for e in nn_err])
    print(f"Wrote {FIG_DIR / 'cp_reconstruction_error.png'}")


if __name__ == "__main__":
    main()
