"""
Fabry–Perot grid-search power tensor generator.

Runs FINESSE over a discrete grid of cavity parameters (LX, mirror R/Rc) and
writes port carrier powers to ``tensors/fp_simulation.npz`` next to this file.

Load example::

    import numpy as np
    d = np.load("data/tensors/fp_simulation.npz", allow_pickle=True)
    # powers shape: (n_LX, n_m1_R, n_m1_Rc, n_m2_R, n_m2_Rc, n_ports)
    p = d["powers"]
    lx, m1_r = d["LX"], d["m1_R"]
"""

from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import finesse
import numpy as np

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = DATA_DIR / "tensors" / "fp_simulation.npz"

NOMINAL_LX = 10.0
NOMINAL_M1_R = 0.9
NOMINAL_M1_RC = -17.0
NOMINAL_M2_R = 0.9
NOMINAL_M2_RC = 20.0
NOMINAL_LASER_P = 1.0

FABRY_PEROT_PORT_ORDER: tuple[str, ...] = (
    "l0.p1.i",
    "l0.p1.o",
    "m1.p1.i",
    "m1.p1.o",
    "m1.p2.i",
    "m1.p2.o",
    "m2.p1.i",
    "m2.p1.o",
    "m2.p2.i",
    "m2.p2.o",
)

AXIS_NAMES = ("LX", "m1_R", "m1_Rc", "m2_R", "m2_Rc")

# Hard search limits (physical / numerical sanity, not stability).
_SEARCH_LIMITS: dict[str, tuple[float, float]] = {
    "LX": (0.1, 80.0),
    "m1_R": (0.05, 0.999),
    "m1_Rc": (-2_000.0, -1.0),
    "m2_R": (0.05, 0.999),
    "m2_Rc": (1.0, 2_000.0),
}

# Where stability does not bound the axis (always stable in range), use this design span.
_DESIGN_SPAN: dict[str, tuple[float, float]] = {
    "m1_R": (0.35, 0.99),
    "m2_R": (0.35, 0.99),
}

_EDGE_MARGIN_FRAC = 0.01  # shrink discovered stability edges 1% toward nominal


def _kat_script(
    lx: float,
    m1_r: float,
    m1_rc: float,
    m2_r: float,
    m2_rc: float,
    laser_p: float = NOMINAL_LASER_P,
) -> str:
    m1_r = min(1.0, float(m1_r))
    m2_r = min(1.0, float(m2_r))
    m1_t = 1.0 - m1_r
    m2_t = 1.0 - m2_r
    return f"""
    l l0 P={laser_p}
    bp roc_l0 l0.p1.o rc
    s s0 l0.p1 m1.p1 0
    m m1 R={m1_r} T={m1_t} Rc={m1_rc}
    s LX m1.p2 m2.p1 L={lx}
    m m2 R={m2_r} T={m2_t} Rc={m2_rc}
    cav cavity1 m1.p2.o priority=0
    noxaxis()
    """


def _attach_detectors(model: finesse.Model) -> None:
    for node in model.optical_network.nodes():
        name = node.replace(".", "_")
        model.parse(f"pd p_{name} {node}")
        model.parse(f"fd f_{name} {node} f=0")
        model.parse(f"bp q_{name} {node} prop=q")


def build_fabry_perot_model(
    lx: float = NOMINAL_LX,
    m1_r: float = NOMINAL_M1_R,
    m1_rc: float = NOMINAL_M1_RC,
    m2_r: float = NOMINAL_M2_R,
    m2_rc: float = NOMINAL_M2_RC,
    laser_p: float = NOMINAL_LASER_P,
) -> finesse.Model:
    model = finesse.Model()
    model.parse(_kat_script(lx, m1_r, m1_rc, m2_r, m2_rc, laser_p))
    model.modes(maxtem=6, modes="even")
    _attach_detectors(model)
    return model


def simulate_pd_vector(
    model: finesse.Model,
    port_order: Sequence[str] = FABRY_PEROT_PORT_ORDER,
) -> tuple[np.ndarray, bool]:
    """Run FINESSE and return (pd watts, success). success requires stable cavity."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = model.run()
    except Exception:
        return np.zeros(len(port_order), dtype=np.float64), False
    if not model.cavity1.is_stable:
        return np.zeros(len(port_order), dtype=np.float64), False
    pd = np.array(
        [float(out[f"p_{node.replace('.', '_')}"]) for node in port_order],
        dtype=np.float64,
    )
    return pd, True


def run_grid_point(
    lx: float,
    m1_r: float,
    m1_rc: float,
    m2_r: float,
    m2_rc: float,
    laser_p: float = NOMINAL_LASER_P,
) -> tuple[np.ndarray, bool]:
    model = build_fabry_perot_model(lx, m1_r, m1_rc, m2_r, m2_rc, laser_p)
    return simulate_pd_vector(model)


def _nominal_params() -> dict[str, float]:
    return {
        "LX": NOMINAL_LX,
        "m1_R": NOMINAL_M1_R,
        "m1_Rc": NOMINAL_M1_RC,
        "m2_R": NOMINAL_M2_R,
        "m2_Rc": NOMINAL_M2_RC,
    }


def _is_stable_at(
    params: dict[str, float],
    laser_p: float = NOMINAL_LASER_P,
) -> bool:
    _, ok = run_grid_point(
        params["LX"],
        params["m1_R"],
        params["m1_Rc"],
        params["m2_R"],
        params["m2_Rc"],
        laser_p,
    )
    return ok


def _is_stable_nominal_slice(
    axis: str,
    value: float,
    nominal: dict[str, float],
    laser_p: float,
) -> bool:
    params = dict(nominal)
    params[axis] = value
    return _is_stable_at(params, laser_p)


def _apply_margin(lo: float, hi: float, nominal: float) -> tuple[float, float]:
    """Pull endpoints slightly inward from the discovered edge toward nominal."""
    if hi <= lo:
        return lo, hi
    span = hi - lo
    m = _EDGE_MARGIN_FRAC * span
    lo_m = lo + m if nominal > lo else lo
    hi_m = hi - m if nominal < hi else hi
    if hi_m <= lo_m:
        mid = 0.5 * (lo + hi)
        eps = 0.01 * max(abs(span), 1e-9)
        return mid - eps, mid + eps
    return lo_m, hi_m


def _bracket_edge(
    axis: str,
    nominal: dict[str, float],
    direction: int,
    laser_p: float,
) -> tuple[float, float, bool, bool]:
    """
    Return (stable_value, trial_value, stable_side_is_lo, found_bracket).
    Walk from nominal along `direction` (+1 or -1) until stability flips.
    """
    lim_lo, lim_hi = _SEARCH_LIMITS[axis]
    nom_val = nominal[axis]
    if not _is_stable_nominal_slice(axis, nom_val, nominal, laser_p):
        raise RuntimeError(f"Nominal {axis}={nom_val} is not stable")

    step = max(abs(nom_val) * 0.05, 0.05)
    last_good = nom_val
    trial = nom_val
    for _ in range(40):
        trial = last_good + direction * step
        trial = max(lim_lo, min(lim_hi, trial))
        if trial == last_good:
            break
        if not _is_stable_nominal_slice(axis, trial, nominal, laser_p):
            return last_good, trial, direction < 0, True
        last_good = trial
        step *= 1.35

    # No instability found within search limits — use limit as edge.
    return last_good, trial, direction < 0, False


def _binary_refine_edge(
    axis: str,
    nominal: dict[str, float],
    stable_val: float,
    unstable_val: float,
    laser_p: float,
) -> float:
    lo, hi = (stable_val, unstable_val) if stable_val < unstable_val else (unstable_val, stable_val)
    stable_at_lo = _is_stable_nominal_slice(axis, lo, nominal, laser_p)
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if _is_stable_nominal_slice(axis, mid, nominal, laser_p):
            if stable_at_lo:
                lo = mid
            else:
                hi = mid
        else:
            if stable_at_lo:
                hi = mid
            else:
                lo = mid
    return lo if stable_at_lo else hi


def discover_stable_edges(
    laser_p: float = NOMINAL_LASER_P,
    nominal: dict[str, float] | None = None,
) -> dict[str, tuple[float, float]]:
    """
    Find local stable interval around nominal for each axis (1D slices).
    Reflectivities use a wide fixed design span when always stable.
    """
    if nominal is None:
        nominal = _nominal_params()

    bounds: dict[str, tuple[float, float]] = {}
    for axis in AXIS_NAMES:
        if axis in _DESIGN_SPAN:
            bounds[axis] = _DESIGN_SPAN[axis]
            print(f"  {axis}: design span {_DESIGN_SPAN[axis]}")
            continue

        good_lo, bad_lo, stable_at_lo, found_lo = _bracket_edge(
            axis, nominal, -1, laser_p
        )
        good_hi, bad_hi, stable_at_hi, found_hi = _bracket_edge(
            axis, nominal, +1, laser_p
        )

        lo = (
            _binary_refine_edge(axis, nominal, good_lo, bad_lo, laser_p)
            if found_lo
            else _SEARCH_LIMITS[axis][0]
        )
        hi = (
            _binary_refine_edge(axis, nominal, good_hi, bad_hi, laser_p)
            if found_hi
            else _SEARCH_LIMITS[axis][1]
        )
        if not found_lo and axis == "m1_Rc":
            # No low-Rc instability found; mirror the span seen toward +Rc (less negative).
            lo = nominal[axis] - 2.5 * abs(hi - nominal[axis])
        if not found_hi and axis == "m2_Rc":
            hi = nominal[axis] + 2.5 * abs(nominal[axis] - lo)
        if lo > hi:
            lo, hi = hi, lo
        lo, hi = _apply_margin(lo, hi, nominal[axis])
        bounds[axis] = (lo, hi)
        print(f"  {axis}: stable edge [{lo:.6g}, {hi:.6g}]")

    return bounds


def axes_from_edges(
    bounds: dict[str, tuple[float, float]],
    points: int,
) -> dict[str, np.ndarray]:
    return {
        name: np.linspace(bounds[name][0], bounds[name][1], points)
        for name in AXIS_NAMES
    }


def suggest_stable_bounds(
    axes: dict[str, np.ndarray],
    laser_p: float = NOMINAL_LASER_P,
    nominal: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """
    Shrink each 1D axis so min/max endpoints are stable at nominal other parameters.
    Interior grid points are unchanged; only endpoints may move inward.
    """
    if nominal is None:
        nominal = _nominal_params()
    out: dict[str, np.ndarray] = {}
    for name, values in axes.items():
        arr = np.asarray(values, dtype=np.float64).copy()
        if arr.size < 2:
            out[name] = arr
            continue
        lo_idx, hi_idx = 0, arr.size - 1
        while lo_idx < hi_idx and not _is_stable_nominal_slice(
            name, arr[lo_idx], nominal, laser_p
        ):
            lo_idx += 1
        while hi_idx > lo_idx and not _is_stable_nominal_slice(
            name, arr[hi_idx], nominal, laser_p
        ):
            hi_idx -= 1
        trimmed = arr[lo_idx : hi_idx + 1]
        if trimmed.size < arr.size:
            print(
                f"  stability trim {name}: {arr.size} -> {trimmed.size} points "
                f"([{arr[0]:.6g}, {arr[-1]:.6g}] -> [{trimmed[0]:.6g}, {trimmed[-1]:.6g}])"
            )
        out[name] = trimmed
    return out


def filter_constant_ports(
    powers: np.ndarray,
    valid_mask: np.ndarray,
    port_names: Sequence[str],
    tol_abs: float = 1e-12,
    tol_rel: float = 1e-9,
) -> tuple[np.ndarray, list[str]]:
    """Drop ports whose power is constant over valid grid cells."""
    active_idx: list[int] = []
    active_names: list[str] = []
    for p, name in enumerate(port_names):
        vals = powers[..., p][valid_mask]
        if vals.size < 2:
            continue
        span = float(np.ptp(vals))
        scale = max(float(np.max(np.abs(vals))), 1.0)
        if span > tol_abs and span / scale > tol_rel:
            active_idx.append(p)
            active_names.append(name)
    if not active_idx:
        return powers[..., :0], []
    return powers[..., active_idx], active_names


def build_default_axes(
    points: int,
    paper_pert: bool = False,
) -> dict[str, np.ndarray]:
    if paper_pert:
        pct = 0.0001
        return {
            "LX": np.linspace(NOMINAL_LX * (1 - pct), NOMINAL_LX * (1 + pct), points),
            "m1_R": np.linspace(NOMINAL_M1_R * (1 - pct), NOMINAL_M1_R * (1 + pct), points),
            "m1_Rc": np.linspace(NOMINAL_M1_RC * (1 - pct), NOMINAL_M1_RC * (1 + pct), points),
            "m2_R": np.linspace(NOMINAL_M2_R * (1 - pct), NOMINAL_M2_R * (1 + pct), points),
            "m2_Rc": np.linspace(NOMINAL_M2_RC * (1 - pct), NOMINAL_M2_RC * (1 + pct), points),
        }
    return {
        "LX": np.linspace(9.0, 11.0, points),
        "m1_R": np.linspace(0.85, 0.95, points),
        "m1_Rc": np.linspace(NOMINAL_M1_RC * 0.9, NOMINAL_M1_RC * 1.1, points),
        "m2_R": np.linspace(0.85, 0.95, points),
        "m2_Rc": np.linspace(NOMINAL_M2_RC * 0.9, NOMINAL_M2_RC * 1.1, points),
    }


def _has_manual_axis_overrides(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name) is not None
        for name in (
            "lx_min",
            "lx_max",
            "m1_r_min",
            "m1_r_max",
            "m1_rc_min",
            "m1_rc_max",
            "m2_r_min",
            "m2_r_max",
            "m2_rc_min",
            "m2_rc_max",
        )
    )


def _apply_axis_overrides(
    axes: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    overrides = {
        "LX": (args.lx_min, args.lx_max),
        "m1_R": (args.m1_r_min, args.m1_r_max),
        "m1_Rc": (args.m1_rc_min, args.m1_rc_max),
        "m2_R": (args.m2_r_min, args.m2_r_max),
        "m2_Rc": (args.m2_rc_min, args.m2_rc_max),
    }
    n = len(next(iter(axes.values())))
    for key, (lo, hi) in overrides.items():
        if lo is not None and hi is not None:
            axes[key] = np.linspace(lo, hi, n)
    return axes


def run_grid(
    axes: dict[str, np.ndarray],
    laser_p: float = NOMINAL_LASER_P,
    max_sims: int | None = None,
    progress_every: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    shape = tuple(len(axes[k]) for k in AXIS_NAMES)
    n_ports = len(FABRY_PEROT_PORT_ORDER)
    powers = np.zeros(shape + (n_ports,), dtype=np.float64)
    valid_mask = np.zeros(shape, dtype=bool)

    grids = [axes[k] for k in AXIS_NAMES]
    total = int(np.prod(shape))
    if max_sims is not None:
        total = min(total, max_sims)

    print(f"Grid shape {shape} -> {int(np.prod(shape))} points", end="")
    if max_sims is not None:
        print(f" (running first {max_sims})", end="")
    print()

    count = 0
    for idx, combo in enumerate(itertools.product(*grids)):
        if max_sims is not None and count >= max_sims:
            break
        lx, m1_r, m1_rc, m2_r, m2_rc = combo
        pd, ok = run_grid_point(lx, m1_r, m1_rc, m2_r, m2_rc, laser_p)
        multi = np.unravel_index(idx, shape)
        valid_mask[multi] = ok
        if ok:
            powers[multi] = pd
        count += 1
        if progress_every and count % progress_every == 0:
            print(f"  {count}/{total} simulations")

    return powers, valid_mask


def save_npz(
    path: Path,
    powers: np.ndarray,
    valid_mask: np.ndarray,
    axes: dict[str, np.ndarray],
    port_names_active: Sequence[str],
    edge_bounds: dict[str, tuple[float, float]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kw: dict = {
        "powers": powers,
        "valid_mask": valid_mask,
        "port_names_active": np.array(port_names_active, dtype=object),
        "port_names_all": np.array(FABRY_PEROT_PORT_ORDER, dtype=object),
        "LX": axes["LX"],
        "m1_R": axes["m1_R"],
        "m1_Rc": axes["m1_Rc"],
        "m2_R": axes["m2_R"],
        "m2_Rc": axes["m2_Rc"],
        "nominal_LX": np.float64(NOMINAL_LX),
        "nominal_m1_R": np.float64(NOMINAL_M1_R),
        "nominal_m1_Rc": np.float64(NOMINAL_M1_RC),
        "nominal_m2_R": np.float64(NOMINAL_M2_R),
        "nominal_m2_Rc": np.float64(NOMINAL_M2_RC),
        "laser_P": np.float64(NOMINAL_LASER_P),
    }
    if edge_bounds is not None:
        for name in AXIS_NAMES:
            lo, hi = edge_bounds[name]
            kw[f"edge_lo_{name}"] = np.float64(lo)
            kw[f"edge_hi_{name}"] = np.float64(hi)
    np.savez(path, **kw)
    print(f"Wrote {path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fabry–Perot grid power tensor generator")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output .npz path")
    p.add_argument("--points", type=int, default=10, help="Points per axis between discovered edges")
    p.add_argument("--paper-pert", action="store_true", help="±0.01%% spans around nominal")
    p.add_argument("--dry-run", action="store_true", help="Print grid only, no FINESSE")
    p.add_argument("--max-sims", type=int, default=None, help="Cap simulations (smoke tests)")
    p.add_argument(
        "--discover-edges",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Find local stable/design edges and grid between them (default: on)",
    )
    p.add_argument("--no-stability-trim", action="store_true", help="Skip endpoint stability trim")
    p.add_argument("--progress-every", type=int, default=500)
    p.add_argument("--lx-min", type=float, default=None)
    p.add_argument("--lx-max", type=float, default=None)
    p.add_argument("--m1-r-min", type=float, default=None)
    p.add_argument("--m1-r-max", type=float, default=None)
    p.add_argument("--m1-rc-min", type=float, default=None)
    p.add_argument("--m1-rc-max", type=float, default=None)
    p.add_argument("--m2-r-min", type=float, default=None)
    p.add_argument("--m2-r-max", type=float, default=None)
    p.add_argument("--m2-rc-min", type=float, default=None)
    p.add_argument("--m2-rc-max", type=float, default=None)
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    edge_bounds: dict[str, tuple[float, float]] | None = None
    if args.paper_pert:
        axes = build_default_axes(args.points, paper_pert=True)
    elif args.discover_edges and not _has_manual_axis_overrides(args):
        print("Discovering local stable / design edges (nominal slice)...")
        edge_bounds = discover_stable_edges()
        axes = axes_from_edges(edge_bounds, args.points)
    else:
        axes = build_default_axes(args.points, paper_pert=False)
    axes = _apply_axis_overrides(axes, args)

    shape = tuple(len(axes[k]) for k in AXIS_NAMES)
    print("Axes:")
    for k in AXIS_NAMES:
        v = axes[k]
        print(f"  {k}: {len(v)} points [{v[0]:.6g}, {v[-1]:.6g}]")
    print(f"Total grid points: {int(np.prod(shape))}")

    if args.dry_run:
        return

    if not args.no_stability_trim and not args.discover_edges:
        print("Stability trim on axis endpoints...")
        axes = suggest_stable_bounds(axes)

    powers, valid_mask = run_grid(
        axes,
        max_sims=args.max_sims,
        progress_every=args.progress_every,
    )

    n_valid = int(valid_mask.sum())
    print(f"Valid simulations: {n_valid} / {valid_mask.size}")

    powers, active_names = filter_constant_ports(
        powers, valid_mask, FABRY_PEROT_PORT_ORDER
    )
    print(f"Active ports ({len(active_names)}): {active_names}")

    save_npz(args.out, powers, valid_mask, axes, active_names, edge_bounds)


if __name__ == "__main__":
    main()
