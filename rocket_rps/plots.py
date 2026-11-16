"""Graphs and slow GIFs of the solved strategy and of matches between bots.

    python -m rocket_rps.plots                                # everything, into figures/
    python -m rocket_rps.plots --replay Adaptive RocketsLate  # one match replay
    python -m rocket_rps.plots --no-gifs                      # still images only
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

from .bots import default_bots
from .game import ROCKET, WATER, play_match
from .solver import load_or_solve

# Colours: one per kind of move, used the same way in every figure.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e3e2de"
C_WATER = "#2a78d6"    # blue
C_ROCKET = "#eb6834"   # orange
C_PLAIN = "#1baf7a"    # aqua
WATER_MAP = LinearSegmentedColormap.from_list("water", ["#f3f6fb", "#a9c8ee", C_WATER, "#123f78"])

FPS = 4            # slow on purpose, so each frame can be read
HOLD_FRAMES = 12   # pause on the last frame before the GIF loops

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "lines.solid_capstyle": "round",
})


def _save_gif(anim, path):
    anim.save(path, writer=PillowWriter(fps=FPS))
    plt.close(anim._fig)
    print("  saved", path)


def _save_png(fig, path):
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  saved", path)


# ---------------------------------------------------------------------------
# 1. The solved strategy through a match
# ---------------------------------------------------------------------------

def strategy_through_match(solution, out_dir, matches=300):
    """Average of Optimal v Optimal: how often each move is played, round by round."""
    rounds, rockets = solution.rounds, solution.rockets
    bots = default_bots(solution)
    a, b = bots[0], default_bots(solution)[0]
    p_rocket = np.zeros(rounds)
    p_water = np.zeros(rounds)
    left = np.zeros(rounds)
    for m in range(matches):
        res = play_match(a, b, rounds, rockets, seed=m, keep_history=True)
        ha = np.array(res.history_a)
        r1 = rockets - np.concatenate([[0], np.cumsum(ha == ROCKET)[:-1]])
        r2 = rockets - np.concatenate([[0], np.cumsum(np.array(res.history_b) == ROCKET)[:-1]])
        n = np.arange(rounds, 0, -1)
        pol = solution.policy[n, r1, r2]
        p_rocket += pol[:, 0]
        p_water += pol[:, 1]
        left += r1
    p_rocket /= matches
    p_water /= matches
    left /= matches
    p_plain = 1 - p_rocket - p_water
    x = np.arange(1, rounds + 1)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.3]})
    for y, c, label in [(p_plain, C_PLAIN, "rock, paper or scissors"), (p_water, C_WATER, "water"),
                        (p_rocket, C_ROCKET, "rocket")]:
        ax.plot(x, y, color=c)
        k = int(rounds * 0.3)
        ax.annotate(label, (x[k], y[k]), xytext=(0, 7), textcoords="offset points",
                    color=INK, fontsize=10, ha="center")
    ax.set_ylim(0, 1)
    ax.set_ylabel("chance of playing it")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title(f"How the solved strategy plays through a match\n"
                 f"Optimal v Optimal, average of {matches} matches", loc="left")
    ax2.plot(x, left, color=C_ROCKET)
    ax2.set_ylabel("rockets left")
    ax2.set_xlabel("round")
    ax2.set_ylim(0, rockets * 1.05)
    ax2.set_xlim(1, rounds)
    fig.tight_layout()
    _save_png(fig, os.path.join(out_dir, "strategy_through_match.png"))


# ---------------------------------------------------------------------------
# 2. Water rate as a heatmap over both rocket counts
# ---------------------------------------------------------------------------

def _water_curve(solution, n):
    """Chance of water against each number of opponent rockets, with n rounds left.

    The solved water rate does not depend on my own rockets at all, so one row
    is the whole picture.
    """
    return solution.policy[n, 0, :, 1]


def water_map(solution, out_dir):
    """One picture of every situation: round on the x-axis, opponent's rockets on the y-axis."""
    rounds, rockets = solution.rounds, solution.rockets
    grid = np.array([_water_curve(solution, n) for n in range(rounds, 0, -1)]).T   # [their rockets, round]
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    im = ax.imshow(grid, origin="lower", cmap=WATER_MAP, vmin=0, vmax=0.5, aspect="auto",
                   extent=[0, rounds, 0, rockets])
    ax.grid(False)
    played = np.arange(0, rounds + 1)
    ax.plot(played, (rounds - played) / 6, color=INK, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.set_ylim(0, rockets)
    ax.text(0.30 * rounds, 0.30 * rockets, "No water needed:\nthe opponent has time to\nfire every rocket at 1 in 6",
            ha="center", va="center", color=MUTED, fontsize=10)
    ax.text(0.80 * rounds, 0.72 * rockets, "Water half the time:\nthe opponent has more rockets\nthan time to use them",
            ha="center", va="center", color="white", fontsize=10)
    ax.annotate("rockets = rounds left ÷ 6", xy=(0.64 * rounds, 0.36 * rounds / 6), xytext=(0.53 * rounds, 0.60 * rockets),
                ha="right", va="center", color=INK, fontsize=10, arrowprops={"arrowstyle": "-", "color": INK, "linewidth": 1})
    ax.set_xlabel("rounds played")
    ax.set_ylabel("rockets the opponent still has")
    ax.set_title("When the solved strategy plays water")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("chance of playing water this round")
    cb.ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    cb.outline.set_visible(False)
    _save_png(fig, os.path.join(out_dir, "water_map.png"))


def water_curve_gif(solution, out_dir):
    """The water rule as a moving curve: it slides left as the rounds run out."""
    rounds, rockets = solution.rounds, solution.rockets
    ns = [n for n in range(rounds, 0, -25)] + [10, 1]
    x = np.arange(rockets + 1)
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=100)
    ax.set_xlim(0, rockets)
    ax.set_ylim(0, 0.56)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("rockets the opponent still has")
    ax.set_ylabel("chance I play water this round")
    title = ax.set_title("When to play water")
    line, = ax.plot([], [], color=C_WATER)
    marker = ax.axvline(0, color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
    label = ax.text(0, 0.535, "", color=INK, fontsize=9, va="center")
    fill = [None]
    fig.tight_layout()

    def draw(i):
        n = ns[min(i, len(ns) - 1)]
        y = _water_curve(solution, n)
        line.set_data(x, y)
        if fill[0] is not None:
            fill[0].remove()
        fill[0] = ax.fill_between(x, 0, y, color=C_WATER, alpha=0.15, linewidth=0)
        t = n / 6
        marker.set_xdata([t, t])
        marker.set_visible(t <= rockets)
        if t <= rockets:
            right = t < 0.6 * rockets
            label.set_text(("  " if right else "") + f"rounds left ÷ 6 = {t:.0f}" + ("" if right else "  "))
            label.set_x(t)
            label.set_ha("left" if right else "right")
        else:
            label.set_text(f"rounds left ÷ 6 = {t:.0f}, more rockets than anyone has  ")
            label.set_x(rockets)
            label.set_ha("right")
        title.set_text(f"When to play water: {n} rounds left")
        return line, title

    _save_gif(FuncAnimation(fig, draw, frames=len(ns) + HOLD_FRAMES, blit=False),
              os.path.join(out_dir, "water_curve.gif"))


# ---------------------------------------------------------------------------
# 3. Replay of one match
# ---------------------------------------------------------------------------

def _replay_data(bot_a, bot_b, rounds, rockets, seed):
    res = play_match(bot_a, bot_b, rounds, rockets, seed=seed, keep_history=True)
    from .game import PAYOFF
    ha, hb = np.array(res.history_a), np.array(res.history_b)
    lead = np.cumsum([PAYOFF[a][b] for a, b in zip(ha, hb)])
    return {
        "res": res, "lead": lead,
        "rockets_a": rockets - np.cumsum(ha == ROCKET), "rockets_b": rockets - np.cumsum(hb == ROCKET),
        "water_a": np.cumsum(ha == WATER), "water_b": np.cumsum(hb == WATER),
    }


def _replay_figure(d, rounds, rockets, dpi=None):
    res = d["res"]
    fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(8.6, 7.4), sharex=True, dpi=dpi,
                                       gridspec_kw={"height_ratios": [2.6, 1.2, 1.2]})
    lim = max(10, float(np.abs(d["lead"]).max())) * 1.12
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_ylim(-lim, lim)
    ax.set_xlim(0, rounds)
    ax.set_ylabel(f"lead for {res.name_a}")
    ax.set_title(f"{res.name_a} v {res.name_b}", loc="left")
    ax.text(0.995, 0.97, f"{res.name_a} ahead", transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=9)
    ax.text(0.995, 0.03, f"{res.name_b} ahead", transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=9)
    ax2.set_ylim(0, rockets * 1.08)
    ax2.set_ylabel("rockets left")
    ax3.set_ylim(0, max(5, d["water_a"].max(), d["water_b"].max()) * 1.12)
    ax3.set_ylabel("water played")
    ax3.set_xlabel("round")
    fig.align_ylabels()
    fig.tight_layout()
    return fig, (ax, ax2, ax3)


def _legend(ax, name_a, name_b):
    ax.figure.legend([plt.Line2D([], [], color=C_WATER), plt.Line2D([], [], color=C_ROCKET)],
                     [name_a, name_b], loc="upper right", frameon=False, ncol=2, fontsize=10,
                     bbox_to_anchor=(0.985, 0.995))


def replay(solution, name_a, name_b, out_dir, seed=1, gif=True):
    """Running score, rockets left and water played for one match."""
    rounds, rockets = solution.rounds, solution.rockets
    bots = {b.name.lower(): b for b in default_bots(solution)}
    other = {b.name.lower(): b for b in default_bots(solution)}   # separate objects for mirror matches
    try:
        a, b = bots[name_a.lower()], other[name_b.lower()]
    except KeyError as e:
        raise SystemExit(f"Unknown bot {e}. Choose from: {', '.join(x.name for x in bots.values())}")
    d = _replay_data(a, b, rounds, rockets, seed)
    x = np.arange(1, rounds + 1)
    stem = os.path.join(out_dir, f"replay_{a.name}_v_{b.name}")

    def draw_all(axes, upto):
        ax, ax2, ax3 = axes
        s = slice(0, upto)
        lines = [ax.plot(x[s], d["lead"][s], color=INK)[0],
                 ax2.plot(x[s], d["rockets_a"][s], color=C_WATER)[0],
                 ax2.plot(x[s], d["rockets_b"][s], color=C_ROCKET)[0],
                 ax3.plot(x[s], d["water_a"][s], color=C_WATER)[0],
                 ax3.plot(x[s], d["water_b"][s], color=C_ROCKET)[0]]
        return lines

    fig, axes = _replay_figure(d, rounds, rockets)
    draw_all(axes, rounds)
    _legend(axes[1], a.name, b.name)
    axes[0].text(0.01, 0.97, f"final lead {d['lead'][-1]:+d}", transform=axes[0].transAxes,
                 ha="left", va="top", fontsize=11, fontweight="bold")
    _save_png(fig, stem + ".png")
    if not gif:
        return

    fig, axes = _replay_figure(d, rounds, rockets, dpi=90)
    _legend(axes[1], a.name, b.name)
    lines = draw_all(axes, 1)
    series = [d["lead"], d["rockets_a"], d["rockets_b"], d["water_a"], d["water_b"]]
    score = axes[0].text(0.01, 0.97, "", transform=axes[0].transAxes, ha="left", va="top",
                         fontsize=11, fontweight="bold")
    steps = np.linspace(rounds / 50, rounds, 50).astype(int)

    def draw(i):
        upto = steps[min(i, len(steps) - 1)]
        for line, y in zip(lines, series):
            line.set_data(x[:upto], y[:upto])
        score.set_text(f"round {upto}: lead {d['lead'][upto - 1]:+d}")
        return lines + [score]

    _save_gif(FuncAnimation(fig, draw, frames=len(steps) + HOLD_FRAMES, blit=False), stem + ".gif")


DEFAULT_REPLAYS = [("Optimal", "RocketsLate"), ("Adaptive", "RocketsLate"),
                   ("Pattern", "BeatLast"), ("Adaptive", "Pattern")]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Graphs and GIFs for Rocket RPS")
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--rockets", type=int, default=100)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-gifs", action="store_true")
    ap.add_argument("--replay", nargs=2, metavar=("BOT_A", "BOT_B"),
                    help="only make the replay of one match between these two bots")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    solution = load_or_solve(args.rounds, args.rockets)
    if args.replay:
        replay(solution, *args.replay, args.out, args.seed, gif=not args.no_gifs)
        return
    strategy_through_match(solution, args.out)
    water_map(solution, args.out)
    if not args.no_gifs:
        water_curve_gif(solution, args.out)
    for a, b in DEFAULT_REPLAYS:
        replay(solution, a, b, args.out, args.seed, gif=not args.no_gifs)


if __name__ == "__main__":
    main()
