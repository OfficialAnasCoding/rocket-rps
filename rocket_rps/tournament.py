"""Round-robin tournament between the bots.

    python -m rocket_rps.tournament                 # full game: 1000 rounds, 100 rockets
    python -m rocket_rps.tournament --matches 1000 --jobs 4   # big run, less noise
    python -m rocket_rps.tournament --rounds 100 --rockets 10
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time

from .bots import default_bots
from .game import play_match
from .solver import load_or_solve


def _play_pairing(job):
    i, j, rounds, rockets, matches, seed = job
    a, b = _BOTS[i], _BOTS[j]
    base = seed * 1_000_003 + (i * len(_BOTS) + j) * 100_003
    return i, j, [play_match(a, b, rounds, rockets, seed=base + m).diff for m in range(matches)]


_BOTS = []


def run_tournament(bots, rounds=1000, rockets=100, matches=20, seed=0, verbose=True, jobs=1):
    """Every bot plays every other bot `matches` times.

    Returns results[(i, j)] = list of score differences (bot i minus bot j).
    With jobs > 1 the pairings are shared between that many processes.
    """
    global _BOTS
    _BOTS = bots
    todo = [(i, j, rounds, rockets, matches, seed)
            for i in range(len(bots)) for j in range(i + 1, len(bots))]
    pool = None
    if jobs > 1:
        import multiprocessing as mp
        try:
            pool = mp.get_context("fork").Pool(jobs)   # fork shares the solved game
        except ValueError:
            print("(parallel runs need a system with fork; running on one process)")
    outputs = pool.imap_unordered(_play_pairing, todo) if pool else map(_play_pairing, todo)
    results = {}
    for i, j, diffs in outputs:
        results[(i, j)] = diffs
        results[(j, i)] = [-d for d in diffs]
        if verbose:
            print(f"  {bots[i].name:>14} v {bots[j].name:<14} mean lead {mean(diffs):+8.1f} "
                  f"± {stderr(diffs):.1f}", flush=True)
    if pool:
        pool.close()
        pool.join()
    return results


def mean(xs):
    return sum(xs) / len(xs)


def stderr(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def summarise(bots, results):
    """One row per bot: average lead per match, match record, best and worst pairing."""
    rows = []
    for i, bot in enumerate(bots):
        all_diffs, wins, draws, losses = [], 0, 0, 0
        pair_means = {}
        for j, other in enumerate(bots):
            if i == j:
                continue
            d = results[(i, j)]
            all_diffs += d
            wins += sum(x > 0 for x in d)
            draws += sum(x == 0 for x in d)
            losses += sum(x < 0 for x in d)
            pair_means[other.name] = mean(d)
        worst = min(pair_means, key=pair_means.get)
        best = max(pair_means, key=pair_means.get)
        # Standard error of the overall average, treating each pairing as its own sample.
        se = math.sqrt(sum(stderr(results[(i, j)]) ** 2 for j in range(len(bots)) if j != i)) / (len(bots) - 1)
        rows.append({
            "bot": bot.name, "avg_lead": mean(all_diffs), "std_error": se, "wins": wins, "draws": draws,
            "losses": losses, "win_rate": wins / len(all_diffs),
            "best_vs": best, "best_lead": pair_means[best],
            "worst_vs": worst, "worst_lead": pair_means[worst],
        })
    rows.sort(key=lambda r: -r["avg_lead"])
    return rows


def print_summary(rows):
    print(f"\n{'#':>2} {'Bot':<14} {'Avg lead':>9} {'± error':>8} {'Won':>6} {'Best against':<22} {'Worst against':<22}")
    for k, r in enumerate(rows, 1):
        best = f"{r['best_vs']} ({r['best_lead']:+.0f})"
        worst = f"{r['worst_vs']} ({r['worst_lead']:+.0f})"
        print(f"{k:>2} {r['bot']:<14} {r['avg_lead']:>+9.1f} {r['std_error']:>8.1f} {r['win_rate']:>6.0%} {best:<22} {worst:<22}")


def print_matrix(bots, results):
    names = [b.name[:9] for b in bots]
    print("\nAverage lead per match (row bot minus column bot):")
    print(" " * 15 + "".join(f"{n:>10}" for n in names))
    for i, a in enumerate(bots):
        cells = "".join(f"{'-':>10}" if i == j else f"{mean(results[(i, j)]):>+10.1f}"
                        for j in range(len(bots)))
        print(f"{a.name:<15}{cells}")


def save_csv(bots, results, rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ranking.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bot"] + [b.name for b in bots])
        for i, a in enumerate(bots):
            w.writerow([a.name] + ["" if i == j else round(mean(results[(i, j)]), 2)
                                   for j in range(len(bots))])
    with open(os.path.join(out_dir, "matrix_std_error.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bot"] + [b.name for b in bots])
        for i, a in enumerate(bots):
            w.writerow([a.name] + ["" if i == j else round(stderr(results[(i, j)]), 2)
                                   for j in range(len(bots))])
    with open(os.path.join(out_dir, "matches.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bot_a", "bot_b", "match", "lead_a"])
        for (i, j), diffs in results.items():
            if i < j:
                for m, d in enumerate(diffs, 1):
                    w.writerow([bots[i].name, bots[j].name, m, d])


def save_heatmap(bots, results, rows, path, title):
    """Heatmap of average leads, bots ordered by final ranking."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed, skipping the heatmap)")
        return
    order = [next(i for i, b in enumerate(bots) if b.name == r["bot"]) for r in rows]
    n = len(order)
    grid = [[0.0 if i == j else mean(results[(i, j)]) for j in order] for i in order]
    lim = max(abs(v) for row in grid for v in row) or 1
    fig, ax = plt.subplots(figsize=(1.0 * n + 3, 0.8 * n + 2.2), dpi=150)
    im = ax.imshow(grid, cmap="RdBu", vmin=-lim, vmax=lim)
    names = [bots[i].name for i in order]
    ax.set_xticks(range(n), names, rotation=40, ha="right")
    ax.set_yticks(range(n), names)
    for a in range(n):
        for b in range(n):
            if a != b:
                v = grid[a][b]
                ax.text(b, a, f"{v:+.0f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.55 * lim else "black")
    ax.set_xlabel("opponent")
    ax.set_ylabel("bot")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, label="average lead per match (points)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rocket RPS round-robin tournament")
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--rockets", type=int, default=100)
    ap.add_argument("--matches", type=int, default=20, help="matches per pairing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1, help="processes to use (speeds up big runs)")
    ap.add_argument("--out", default="results", help="folder for CSV files and the heatmap")
    ap.add_argument("--no-cache", action="store_true", help="do not save the solved game to disk")
    args = ap.parse_args(argv)

    t = time.time()
    solution = load_or_solve(args.rounds, args.rockets, cache_dir=None if args.no_cache else ".cache")
    print(f"Solution ready ({time.time() - t:.1f}s). Playing {args.matches} matches per pairing "
          f"of {args.rounds} rounds with {args.rockets} rockets each.\n")
    bots = default_bots(solution)
    results = run_tournament(bots, args.rounds, args.rockets, args.matches, args.seed, jobs=args.jobs)
    rows = summarise(bots, results)
    print_matrix(bots, results)
    print_summary(rows)
    save_csv(bots, results, rows, args.out)
    save_heatmap(bots, results, rows, os.path.join(args.out, "heatmap.png"),
                 f"Rocket RPS: {args.rounds} rounds, {args.rockets} rockets, "
                 f"{args.matches} matches per pairing")
    print("\nEach '± error' is one standard error: the true average is very likely within "
          "about twice that of the number shown.")
    print(f"\nSaved ranking.csv, matrix.csv, matrix_std_error.csv, matches.csv and heatmap.png in {args.out}/ "
          f"({time.time() - t:.0f}s in total)")


if __name__ == "__main__":
    main()
