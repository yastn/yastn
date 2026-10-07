"""Benchmark cuSOLVER SVD drivers behind ``torch.linalg.svd`` on square matrices.

The matrices are synthetic but have an exactly known singular spectrum:

    A = Q diag(s) P^H,   Q, P Haar-random orthogonal/unitary,

with ``s`` normalized to ``s[0] == 1`` and decaying either algebraically
(``1/i**p``) or exponentially (``exp(-i/tau)``).  Since the spectrum is known by
construction, every timing is paired with an accuracy number, which matters:
``gesvdj`` is iterative and ``gesvda`` is an approximate (tall-skinny) solver,
so "fastest" is only meaningful next to "accurate enough".

Sizes follow the shapes that show up in PEPS/CTMRG contractions, parametrized by
the bond dimension ``D``: ``L = D**2``, ``2 D**2``, ``3 D**2`` (corner/half
matrices) and ``L = D**4`` (double-layer objects).

The headline output is, per spectrum, the crossover size where ``gesvd`` becomes
faster than ``gesvdj``.

Usage
-----
    python autotune/bench_svd_drivers.py                    # full sweep, float64
    python autotune/bench_svd_drivers.py --series D2 2D2 3D2
    python autotune/bench_svd_drivers.py --dtype complex128 --max-seconds 5
    python autotune/bench_svd_drivers.py --d-max 9 --repeats 5 --out results.csv

Large sizes are skipped automatically when they do not fit in GPU memory, and a
driver is retired from the rest of a sweep once a single call exceeds
``--max-seconds`` (its cost is monotone in ``L``, so larger sizes would only be
worse).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field

import torch

# cuSOLVER drivers exposed by torch.linalg.svd; "default" = let torch choose.
DRIVERS = ("gesvd", "gesvdj", "gesvda", "default")

NO_CAP = 1 << 30

DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}

# name -> (label, s(i) for i = 1..L, before normalization to s[0] = 1)
SPECTRA = {
    "inv_i": ("1/i", lambda i: 1.0 / i),
    "inv_i2": ("1/i^2", lambda i: 1.0 / i**2),
    "inv_i3": ("1/i^3", lambda i: 1.0 / i**3),
    "exp_500": ("exp(-i/500)", lambda i: torch.exp(-i / 500.0)),
    "exp_200": ("exp(-i/200)", lambda i: torch.exp(-i / 200.0)),
}

SERIES = {
    "D2": ("L=D^2", lambda D: D**2),
    "2D2": ("L=2D^2", lambda D: 2 * D**2),
    "3D2": ("L=3D^2", lambda D: 3 * D**2),
    "D4": ("L=D^4", lambda D: D**4),
}


@dataclass
class Row:
    series: str
    D: int
    L: int
    spectrum: str
    driver: str
    status: str = "ok"
    t_med: float = float("nan")
    t_min: float = float("nan")
    t_max: float = float("nan")
    reps: int = 0
    s_abs_err: float = float("nan")   # max_i |s_i - s_i^exact|
    s_rel_err: float = float("nan")   # max relative error over s_i > rel_floor
    note: str = ""


OOM = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)

REAL_OF = {torch.float32: torch.float32, torch.float64: torch.float64,
           torch.complex64: torch.float32, torch.complex128: torch.float64}


def exact_spectrum(L: int, name: str, device, dtype: torch.dtype) -> torch.Tensor:
    """Normalized singular values, descending, with s[0] == 1."""
    rdt = REAL_OF[dtype]
    i = torch.arange(1, L + 1, device=device, dtype=rdt)
    s = SPECTRA[name][1](i)
    return s / s[0]


def haar(L: int, device, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Haar-distributed orthogonal/unitary matrix via QR of a Gaussian."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    A = torch.randn(L, L, generator=g, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    del A
    # absorb the phases of diag(R) into Q, otherwise Q is not Haar-distributed
    d = torch.diagonal(R)
    Q *= (d / d.abs()).unsqueeze(0)
    del R, d
    return Q


def build_matrix(L: int, spectrum: str, device, dtype: torch.dtype, seed: int):
    """A = Q diag(s) P^H with the requested spectrum; returns (A, s_exact)."""
    s = exact_spectrum(L, spectrum, device, dtype)
    Q = haar(L, device, dtype, seed)
    Q *= s.to(dtype).unsqueeze(0)           # Q diag(s), column scaling
    P = haar(L, device, dtype, seed + 1)
    A = Q @ P.conj().mT
    del Q, P
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return A, s


def plan(args, dtype: torch.dtype, budget: float):
    """Decide which (series, D) actually run, before touching the GPU.

    Three cutoffs, applied in this order:

    1. ``--d-max`` / ``--d-max-series`` -- an explicit cap on D, per series.
       ``L = D**4`` grows so fast that the useful D range is much shorter there
       than for ``L = D**2``; on a large-memory HPC node nothing else would stop
       it before a single SVD costs tens of minutes.
    2. ``--l-max`` -- an explicit cap on the matrix dimension.
    3. the device-memory budget (``--mem-frac``).

    Returns ``(sizes, cuts, l_max)``.
    """
    per_d_max = {ser: args.d_max for ser in args.series}
    for kv in args.d_max_series:
        ser, _, val = kv.partition("=")
        if ser not in SERIES:
            raise SystemExit(f"--d-max-series: unknown series {ser!r}, pick from {list(SERIES)}")
        per_d_max[ser] = int(val)

    item = torch.tensor([], dtype=dtype).element_size()
    l_mem = int(math.sqrt(budget / (5 * item))) if math.isfinite(budget) else NO_CAP
    l_max = min(args.l_max or NO_CAP, l_mem)

    sizes, cuts = [], []
    for ser in args.series:
        for D in range(args.d_min, args.d_max + 1):
            L = SERIES[ser][1](D)
            if D > per_d_max[ser]:
                cuts.append((ser, D, L, "cut-dmax", f"D > --d-max-series {ser}={per_d_max[ser]}"))
            elif L > l_max:
                if l_max == l_mem:
                    cuts.append((ser, D, L, "cut-mem", f"~{bytes_needed(L, dtype) / 2**30:.1f} GiB"
                                                       f" > budget {budget / 2**30:.1f} GiB"))
                else:
                    cuts.append((ser, D, L, "cut-lmax", f"L > --l-max {args.l_max}"))
            else:
                sizes.append((ser, D, L))
    sizes.sort(key=lambda x: x[2])
    return sizes, cuts, l_max


def print_plan(sizes, cuts, l_max, per_series_note=""):
    print("size plan" + (f"  (L <= {l_max})" if l_max < NO_CAP else "") + per_series_note)
    for ser in SERIES:
        run = [(D, L) for s_, D, L in sizes if s_ == ser]
        cut = [(D, L, why, note) for s_, D, L, why, note in cuts if s_ == ser]
        if not run and not cut:
            continue
        head = f"  {ser:4s} {SERIES[ser][0]:8s} "
        if run:
            head += f"D={run[0][0]}..{run[-1][0]:<3d} L={run[0][1]}..{run[-1][1]}"
        else:
            head += "nothing to run"
        if cut:
            head += f"   | cut D>={cut[0][0]} ({cut[0][2]}: {cut[0][3]})"
        print(head)
    print()


def bytes_needed(L: int, dtype: torch.dtype) -> int:
    """Rough peak device memory: A, U, V, an internal copy of A + workspace."""
    return 5 * L * L * torch.tensor([], dtype=dtype).element_size()


def time_svd(A: torch.Tensor, driver: str | None, repeats: int, warmup: int):
    """Median/min/max time (s) of one SVD -- device time on CUDA -- and its singular values."""
    cuda = A.is_cuda
    ts = []
    S = None
    for k in range(warmup + repeats):
        if cuda:
            torch.cuda.synchronize()
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            ev0.record()
            U, S_, Vh = torch.linalg.svd(A, full_matrices=False, driver=driver)
            ev1.record()
            torch.cuda.synchronize()
            dt = ev0.elapsed_time(ev1) * 1e-3
        else:
            t0 = time.perf_counter()
            U, S_, Vh = torch.linalg.svd(A, full_matrices=False)
            dt = time.perf_counter() - t0
        if k >= warmup:
            ts.append(dt)
        S = S_
        del U, Vh, S_
    ts.sort()
    med = ts[len(ts) // 2] if len(ts) % 2 else 0.5 * (ts[len(ts) // 2 - 1] + ts[len(ts) // 2])
    return med, ts[0], ts[-1], S


def accuracy(S: torch.Tensor, s_exact: torch.Tensor, rel_floor: float):
    d = (S.double() - s_exact.double()).abs()
    abs_err = d.max().item()
    mask = s_exact.double() > rel_floor
    rel_err = (d[mask] / s_exact.double()[mask]).max().item() if mask.any() else float("nan")
    return abs_err, rel_err


def sweep(args) -> list[Row]:
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    if device.type == "cuda":
        budget = args.mem_frac * torch.cuda.get_device_properties(device).total_memory
    else:
        budget = float("inf")

    sizes, cuts, l_max = plan(args, dtype, budget)
    print_plan(sizes, cuts, l_max)
    if args.plan_only:
        return []

    rows: list[Row] = []
    for ser, D, L, why, note in cuts:                  # keep the cutoffs in the CSV
        for spec in args.spectra:
            for drv in args.drivers:
                rows.append(Row(ser, D, L, spec, drv, status=why, note=note))

    retired: dict[tuple[str, str], int] = {}   # (spectrum, driver) -> L where it blew the budget
    last: dict[tuple[str, str], tuple[int, float]] = {}   # (spectrum, driver) -> (L, t) measured

    # one-time warmup: the first call into each cuSOLVER driver pays for handle
    # creation and workspace allocation, which would otherwise land on L = D^2 at D = 5
    if device.type == "cuda":
        W = torch.randn(512, 512, device=device, dtype=dtype)
        for drv in args.drivers:
            try:
                torch.linalg.svd(W, full_matrices=False, driver=None if drv == "default" else drv)
            except RuntimeError:
                pass
        del W
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    for series, D, L in sizes:
        need = bytes_needed(L, dtype)
        if need > budget:
            for spec in args.spectra:
                for drv in args.drivers:
                    rows.append(Row(series, D, L, spec, drv, status="skip-mem",
                                    note=f"needs ~{need / 2**30:.1f} GiB > budget {budget / 2**30:.1f} GiB"))
            print(f"[skip] {series} D={D} L={L}: ~{need / 2**30:.1f} GiB needed, "
                  f"budget {budget / 2**30:.1f} GiB", flush=True)
            continue

        # small matrices are launch-overhead dominated (noisy) -> average more of them;
        # huge ones cost minutes per call -> one timed call is all we can afford
        if L <= 512:
            reps, wu = 2 * args.repeats, args.warmup
        elif L <= args.big_L:
            reps, wu = args.repeats, args.warmup
        else:
            reps, wu = max(1, args.repeats // 3), 0
        for spec in args.spectra:
            try:
                A, s_exact = build_matrix(L, spec, device, dtype, args.seed)
            except OOM as e:
                for drv in args.drivers:
                    rows.append(Row(series, D, L, spec, drv, status="oom-build", note=str(e)[:80]))
                torch.cuda.empty_cache()
                print(f"[skip] {series} D={D} L={L} {spec}: OOM while building", flush=True)
                continue

            for drv in args.drivers:
                if (spec, drv) in retired:
                    rows.append(Row(series, D, L, spec, drv, status="retired",
                                    note=f"exceeded {args.max_seconds}s at L={retired[(spec, drv)]}"))
                    continue
                # extrapolate O(L^3) from the last measured size: refuse to *start* a
                # call that is predicted to blow the budget, instead of learning it the
                # expensive way (one gesvdj call at L = 12^4 would run for ~an hour)
                if (spec, drv) in last:
                    L0, t0 = last[(spec, drv)]
                    pred = t0 * (L / L0) ** 3
                    if L > L0 and pred > args.max_seconds:
                        retired[(spec, drv)] = L0
                        rows.append(Row(series, D, L, spec, drv, status="cut-time",
                                        note=f"predicted ~{pred:.0f}s from {t0:.2f}s at L={L0}"))
                        print(f"  [cut-time] {drv} for {spec} at L={L}: "
                              f"predicted ~{pred:.0f}s > {args.max_seconds}s", flush=True)
                        continue
                try:
                    med, lo, hi, S = time_svd(A, None if drv == "default" else drv, reps, wu)
                except OOM as e:
                    rows.append(Row(series, D, L, spec, drv, status="oom", note=str(e)[:80]))
                    torch.cuda.empty_cache()
                    continue
                except RuntimeError as e:                 # e.g. driver rejected for this shape
                    rows.append(Row(series, D, L, spec, drv, status="error", note=str(e)[:120]))
                    torch.cuda.empty_cache()
                    continue
                abs_err, rel_err = accuracy(S, s_exact, args.rel_floor)
                del S
                rows.append(Row(series, D, L, spec, drv, "ok", med, lo, hi, reps, abs_err, rel_err))
                last[(spec, drv)] = (L, med)
                print(f"  {series:5s} D={D:2d} L={L:6d} {spec:8s} {drv:8s} "
                      f"{med * 1e3:10.2f} ms   |ds|={abs_err:.2e} rel={rel_err:.2e}", flush=True)
                if med > args.max_seconds:
                    retired[(spec, drv)] = L
                    print(f"  [retire] {drv} for {spec}: {med:.1f}s > {args.max_seconds}s", flush=True)
            del A, s_exact
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def crossover(rows: list[Row], fast: str, slow: str, spectrum: str):
    """Where does `slow` (gesvd) overtake `fast` (gesvdj)?  Interpolate log t vs log L."""
    t = {}
    for r in rows:
        if r.spectrum == spectrum and r.status == "ok" and r.driver in (fast, slow):
            t.setdefault(r.L, {})[r.driver] = r.t_med
    pts = sorted((L, v[fast], v[slow]) for L, v in t.items() if fast in v and slow in v)
    if len(pts) < 2:
        return None
    ratios = [(L, math.log(tf / ts)) for L, tf, ts in pts]   # >0 -> gesvdj slower
    # last sign change from "gesvdj faster" to "gesvdj slower"
    cross, bracket = None, None
    for (L0, r0), (L1, r1) in zip(ratios, ratios[1:]):
        if r0 <= 0.0 < r1:
            w = -r0 / (r1 - r0)
            cross = math.exp(math.log(L0) + w * (math.log(L1) - math.log(L0)))
            bracket = (L0, L1)
    if cross is None:
        # monotone regime: one driver wins everywhere in the sampled range
        winner = fast if ratios[-1][1] < 0 else slow
        return {"spectrum": spectrum, "crossover_L": None, "winner_everywhere": winner,
                "L_range": [pts[0][0], pts[-1][0]]}
    return {"spectrum": spectrum, "crossover_L": round(cross),
            "crossover_D_as_D2": round(math.sqrt(cross)), "crossover_D_as_D4": round(cross**0.25),
            "bracket_L": list(bracket), "L_range": [pts[0][0], pts[-1][0]],
            "n_sign_changes": sum(1 for (_, a), (_, b) in zip(ratios, ratios[1:]) if (a <= 0) != (b <= 0))}


def report(rows: list[Row], args):
    print("\n" + "=" * 96)
    print("median time [ms] per driver")
    print("=" * 96)
    Ls = sorted({r.L for r in rows})
    for spec in args.spectra:
        print(f"\n--- spectrum {SPECTRA[spec][0]} ---")
        print(f"{'L':>7} {'series':>7} " + "".join(f"{d:>12}" for d in args.drivers) + "   best")
        for L in Ls:
            sel = [r for r in rows if r.L == L and r.spectrum == spec]
            if not sel:
                continue
            by = {r.driver: r for r in sel}
            cells = ""
            for d in args.drivers:
                r = by.get(d)
                cells += f"{r.t_med * 1e3:12.2f}" if r and r.status == "ok" else f"{(r.status if r else '-'):>12}"
            ok = [(r.t_med, d) for d, r in by.items() if r.status == "ok" and d != "default"]
            best = min(ok)[1] if ok else "-"
            print(f"{L:7d} {sel[0].series:>7} {cells}   {best}")

    print("\n" + "=" * 96)
    print(f"crossover  gesvdj -> gesvd   (below: gesvdj faster, above: gesvd faster)")
    print("=" * 96)
    out = []
    for spec in args.spectra:
        c = crossover(rows, "gesvdj", "gesvd", spec)
        out.append(c)
        if c is None:
            print(f"{SPECTRA[spec][0]:>12}: not enough data")
        elif c["crossover_L"] is None:
            print(f"{SPECTRA[spec][0]:>12}: no crossover in L in {c['L_range']}, "
                  f"{c['winner_everywhere']} faster throughout")
        else:
            print(f"{SPECTRA[spec][0]:>12}: L* ~ {c['crossover_L']:6d}  "
                  f"(bracketed by L={c['bracket_L'][0]}..{c['bracket_L'][1]}; "
                  f"D~{c['crossover_D_as_D2']} if L=D^2, D~{c['crossover_D_as_D4']} if L=D^4)")

    print("\n" + "=" * 96)
    print("worst singular-value error per driver (max over all sizes/spectra)")
    print("=" * 96)
    for d in args.drivers:
        sel = [r for r in rows if r.driver == d and r.status == "ok"]
        if sel:
            print(f"{d:>10}: max |ds| = {max(r.s_abs_err for r in sel):.3e}   "
                  f"max rel (s > {args.rel_floor:g}) = {max(r.s_rel_err for r in sel):.3e}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float64", choices=sorted(DTYPES))
    p.add_argument("--series", nargs="+", default=list(SERIES), choices=list(SERIES))
    p.add_argument("--spectra", nargs="+", default=list(SPECTRA), choices=list(SPECTRA))
    p.add_argument("--drivers", nargs="+", default=list(DRIVERS), choices=list(DRIVERS))
    p.add_argument("--d-min", type=int, default=5)
    p.add_argument("--d-max", type=int, default=12, help="largest D, all series")
    p.add_argument("--d-max-series", nargs="*", default=[], metavar="SERIES=D",
                   help="per-series cap on D, e.g. --d-max-series D4=9 (L=D^4 grows "
                        "as D^4: on a big node nothing else stops it in time)")
    p.add_argument("--l-max", type=int, default=0,
                   help="cap on the matrix dimension L; 0 = only the memory budget")
    p.add_argument("--plan-only", action="store_true",
                   help="print which sizes would run, then exit (no GPU work)")
    p.add_argument("--repeats", type=int, default=5, help="timed calls per point")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--big-L", type=int, default=2000, help="above this L, use repeats//3")
    p.add_argument("--max-seconds", type=float, default=20.0,
                   help="retire a driver once one call is slower than this")
    p.add_argument("--mem-frac", type=float, default=0.75, help="fraction of device memory to use")
    p.add_argument("--rel-floor", type=float, default=1e-12,
                   help="ignore relative error of singular values below this")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="autotune/svd_drivers.csv")
    args = p.parse_args(argv)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("CUDA not available")
    dev = torch.device(args.device)
    name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
    print(f"torch {torch.__version__} (cuda {torch.version.cuda}) on {name}")
    print(f"dtype={args.dtype} series={args.series} drivers={args.drivers}\n")

    t0 = time.perf_counter()
    rows = sweep(args)
    if not rows:
        return
    cross = report(rows, args)
    print(f"\ntotal sweep time {time.perf_counter() - t0:.1f} s")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["series", "D", "L", "spectrum", "driver", "status", "t_med_s",
                        "t_min_s", "t_max_s", "reps", "s_abs_err", "s_rel_err", "note"])
            for r in rows:
                w.writerow([r.series, r.D, r.L, r.spectrum, r.driver, r.status, r.t_med,
                            r.t_min, r.t_max, r.reps, r.s_abs_err, r.s_rel_err, r.note])
        with open(args.out.rsplit(".", 1)[0] + "_crossover.json", "w") as f:
            json.dump({"device": name, "torch": torch.__version__, "dtype": args.dtype,
                       "crossover_gesvdj_gesvd": cross}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
