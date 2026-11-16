"""Exact solution of Rocket RPS by backward induction.

State: (n, r1, r2) = rounds left, my rockets, opponent's rockets.
V[n, r1, r2] is my expected final lead (my points minus theirs) from that
state when both players play optimally. V[0] = 0, and

    V[n, r1, r2] = value of the one-round zero-sum game whose entries are
                   (points this round) + V[n-1, state after the round].

Rock, paper and scissors are interchangeable, so there is always an optimal
strategy that plays them with equal probability. That reduces every round to
a 3x3 game between the options PLAIN (a uniformly random rock/paper/scissors),
ROCKET and WATER, which is solved in closed form for all states at once.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .game import PLAIN_POINTS, ROCKET_POINTS, WATER_POINTS

EPS = 1e-9
P, K, W = 0, 1, 2  # reduced options: plain, rocket, water

_PAIRS = [(0, 1), (0, 2), (1, 2)]


def reduced_matrix(Vprev: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the 3x3 game for every (r1, r2) from the previous layer of V.

    Returns (M, row_ok, col_ok): M has shape (R1+1, R2+1, 3, 3); row_ok / col_ok
    say whether the rocket option exists for the row / column player.
    """
    n1, n2 = Vprev.shape
    V00 = Vprev                                  # nobody fires
    V10 = np.zeros_like(Vprev); V10[1:, :] = Vprev[:-1, :]     # I fire
    V01 = np.zeros_like(Vprev); V01[:, 1:] = Vprev[:, :-1]     # they fire
    V11 = np.zeros_like(Vprev); V11[1:, 1:] = Vprev[:-1, :-1]  # both fire
    M = np.empty((n1, n2, 3, 3))
    M[..., P, P] = V00                    # plain v plain averages to 0
    M[..., P, K] = -ROCKET_POINTS + V01
    M[..., P, W] = PLAIN_POINTS + V00
    M[..., K, P] = ROCKET_POINTS + V10
    M[..., K, K] = V11
    M[..., K, W] = -WATER_POINTS + V10
    M[..., W, P] = -PLAIN_POINTS + V00
    M[..., W, K] = WATER_POINTS + V01
    M[..., W, W] = V00
    row_ok = np.ones((n1, n2, 3), bool); row_ok[0, :, K] = False
    col_ok = np.ones((n1, n2, 3), bool); col_ok[:, 0, K] = False
    return M, row_ok, col_ok


def solve_3x3(M: np.ndarray, row_ok: np.ndarray, col_ok: np.ndarray):
    """Solve a batch of 3x3 zero-sum games (row player maximises).

    Tries every possible support (all three options, each pair, each single
    option) and keeps the first one that passes the equilibrium checks.
    Returns (value, row strategy, column strategy, solved mask).
    """
    shape = M.shape[:-2]
    val = np.zeros(shape)
    x = np.zeros(shape + (3,))
    y = np.zeros(shape + (3,))
    done = np.zeros(shape, bool)

    def accept(ok, v, xs, ys):
        nonlocal done
        ok = ok & ~done
        if ok.any():
            val[ok] = v[ok]
            x[ok] = xs[ok]
            y[ok] = ys[ok]
            done |= ok

    with np.errstate(divide="ignore", invalid="ignore"):
        # --- all three options used by both players (adjugate formula) ---
        a, b, c = M[..., 0, :], M[..., 1, :], M[..., 2, :]
        adj = np.stack([np.cross(b, c), np.cross(c, a), np.cross(a, b)], axis=-1)
        # adj = adjugate(M): M @ adj = det(M) * I
        s = adj.sum(axis=(-1, -2))
        xs = adj.sum(axis=-2) / s[..., None]     # 1^T adj / s
        ys = adj.sum(axis=-1) / s[..., None]     # adj 1 / s
        det = np.einsum("...i,...i->...", a, np.cross(b, c))
        v = det / s
        ok = (np.abs(s) > 1e-12) & row_ok.all(-1) & col_ok.all(-1)
        ok &= (xs >= -EPS).all(-1) & (ys >= -EPS).all(-1)
        accept(ok, v, np.clip(xs, 0, None), np.clip(ys, 0, None))

        # --- two options each ---
        for i1, i2 in _PAIRS:
            i3 = 3 - i1 - i2
            for j1, j2 in _PAIRS:
                j3 = 3 - j1 - j2
                A, B = M[..., i1, j1], M[..., i1, j2]
                C, D = M[..., i2, j1], M[..., i2, j2]
                den = A - B - C + D
                p = (D - C) / den            # probability of row i1
                q = (D - B) / den            # probability of column j1
                v = (A * D - B * C) / den
                ok = (np.abs(den) > 1e-12) & (p >= -EPS) & (p <= 1 + EPS) & (q >= -EPS) & (q <= 1 + EPS)
                ok &= row_ok[..., i1] & row_ok[..., i2] & col_ok[..., j1] & col_ok[..., j2]
                # the unused row must not do better, the unused column must not do worse
                row3 = q * M[..., i3, j1] + (1 - q) * M[..., i3, j2]
                col3 = p * M[..., i1, j3] + (1 - p) * M[..., i2, j3]
                ok &= ~row_ok[..., i3] | (row3 <= v + EPS)
                ok &= ~col_ok[..., j3] | (col3 >= v - EPS)
                xs = np.zeros(shape + (3,)); xs[..., i1] = p; xs[..., i2] = 1 - p
                ys = np.zeros(shape + (3,)); ys[..., j1] = q; ys[..., j2] = 1 - q
                accept(ok, v, np.clip(xs, 0, 1), np.clip(ys, 0, 1))

        # --- pure saddle points ---
        Mr = np.where(col_ok[..., None, :], M, np.inf)    # ignore missing columns in row minima
        Mc = np.where(row_ok[..., :, None], M, -np.inf)   # ignore missing rows in column maxima
        row_min = Mr.min(-1)
        col_max = Mc.max(-2)
        for i in range(3):
            for j in range(3):
                v = M[..., i, j]
                ok = row_ok[..., i] & col_ok[..., j]
                ok &= (v <= row_min[..., i] + EPS) & (v >= col_max[..., j] - EPS)
                xs = np.zeros(shape + (3,)); xs[..., i] = 1
                ys = np.zeros(shape + (3,)); ys[..., j] = 1
                accept(ok, v, xs, ys)
    return val, x, y, done


def _lp_fallback(M, row_ok, col_ok):
    """Solve one 3x3 game by linear programming (rare degenerate cases)."""
    from scipy.optimize import linprog

    rows = [i for i in range(3) if row_ok[i]]
    cols = [j for j in range(3) if col_ok[j]]

    def lp(G):
        n, m = G.shape
        c = np.zeros(n + 1); c[-1] = -1
        res = linprog(c, A_ub=np.hstack([-G.T, np.ones((m, 1))]), b_ub=np.zeros(m),
                      A_eq=np.hstack([np.ones((1, n)), [[0]]]), b_eq=[1],
                      bounds=[(0, None)] * n + [(None, None)], method="highs")
        return res.x[-1], res.x[:n]

    G = M[np.ix_(rows, cols)]
    v, xr = lp(G)
    _, yc = lp(-G.T)
    x = np.zeros(3); x[rows] = xr
    y = np.zeros(3); y[cols] = yc
    return v, x, y


def solve_layer(Vprev: np.ndarray):
    """Solve every state with one more round left than Vprev."""
    M, row_ok, col_ok = reduced_matrix(Vprev)
    v, x, y, done = solve_3x3(M, row_ok, col_ok)
    for idx in zip(*np.nonzero(~done)):
        v[idx], x[idx], y[idx] = _lp_fallback(M[idx], row_ok[idx], col_ok[idx])
    return v, x, int((~done).sum())


@dataclass
class Solution:
    """The solved game: values and optimal strategies for every state."""

    rounds: int
    rockets: int
    V: np.ndarray        # V[n, r1, r2], n = 0..rounds
    policy: np.ndarray   # policy[n, r1, r2] = (P(rocket), P(water)), n = 1..rounds

    def value(self, n: int, r1: int, r2: int) -> float:
        return float(self.V[n, r1, r2])

    def strategy(self, n: int, r1: int, r2: int) -> dict[str, float]:
        """Optimal move probabilities with n rounds left, r1 my rockets, r2 theirs."""
        pk, pw = (float(t) for t in self.policy[n, r1, r2])
        plain = max(0.0, 1 - pk - pw) / 3
        return {"rock": plain, "paper": plain, "scissors": plain, "rocket": pk, "water": pw}


def solve(rounds: int = 1000, rockets: int = 100, verbose: bool = False) -> Solution:
    """Solve the whole game. About 10 million states for the default size."""
    V = np.zeros((rounds + 1, rockets + 1, rockets + 1))
    policy = np.zeros((rounds + 1, rockets + 1, rockets + 1, 2), np.float32)
    fallbacks = 0
    for n in range(1, rounds + 1):
        V[n], x, f = solve_layer(V[n - 1])
        policy[n, ..., 0] = x[..., K]
        policy[n, ..., 1] = x[..., W]
        fallbacks += f
        if verbose and n % 100 == 0:
            print(f"  solved {n}/{rounds} rounds", flush=True)
    if verbose and fallbacks:
        print(f"  ({fallbacks} states needed the linear-programming fallback)")
    return Solution(rounds, rockets, V, policy)


def load_or_solve(rounds: int = 1000, rockets: int = 100, cache_dir: str | None = ".cache",
                  verbose: bool = True) -> Solution:
    """Solve the game, reusing a saved solution from cache_dir when there is one."""
    path = os.path.join(cache_dir, f"solution_{rounds}_{rockets}.npz") if cache_dir else None
    if path and os.path.exists(path):
        data = np.load(path)
        return Solution(rounds, rockets, data["V"], data["policy"])
    if verbose:
        print(f"Solving Rocket RPS for {rounds} rounds and {rockets} rockets each...", flush=True)
    sol = solve(rounds, rockets, verbose=verbose)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(path, V=sol.V, policy=sol.policy)
    return sol
