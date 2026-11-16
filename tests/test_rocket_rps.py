import math

import numpy as np
import pytest

from rocket_rps.bots import OptimalBot, default_bots
from rocket_rps.game import (MOVES, PAPER, PAYOFF, ROCK, ROCKET, SCISSORS, WATER,
                             play_match, points)
from rocket_rps.solver import solve


def test_scoring_rules():
    assert points(ROCK, SCISSORS) == (1, 0)
    assert points(PAPER, SCISSORS) == (0, 1)
    assert points(ROCKET, PAPER) == (3, 0)
    assert points(WATER, ROCKET) == (2, 0)
    assert points(WATER, ROCK) == (0, 1)
    for m in MOVES:
        assert points(m, m) == (0, 0)
    assert all(PAYOFF[a][b] == -PAYOFF[b][a] for a in MOVES for b in MOVES)


def test_one_round_equilibrium():
    s = solve(1, 1).strategy(1, 1, 1)
    assert s["rocket"] == pytest.approx(1 / 6, abs=1e-6)
    assert s["water"] == pytest.approx(1 / 2, abs=1e-6)
    assert s["rock"] == pytest.approx(1 / 9, abs=1e-6)


def test_hand_worked_values():
    sol = solve(3, 1)
    assert sol.value(1, 1, 0) == pytest.approx(1 / 2)
    assert sol.value(2, 1, 0) == pytest.approx(11 / 12)
    assert sol.strategy(2, 0, 1)["water"] == pytest.approx(5 / 12, abs=1e-6)
    assert sol.value(3, 1, 0) == pytest.approx(1.2638888889)


def test_game_is_fair_and_antisymmetric():
    sol = solve(60, 8)
    assert np.allclose(sol.V, -sol.V.transpose(0, 2, 1), atol=1e-9)
    assert sol.value(60, 8, 8) == pytest.approx(0, abs=1e-9)


def full_game_value(V_prev, r1, r2):
    """Solve one state as the full 5x5 game by linear programming."""
    from scipy.optimize import linprog
    rows = [m for m in MOVES if m != ROCKET or r1 > 0]
    cols = [m for m in MOVES if m != ROCKET or r2 > 0]
    G = np.array([[PAYOFF[a][b] + V_prev[r1 - (a == ROCKET), r2 - (b == ROCKET)] for b in cols]
                  for a in rows])
    c = np.zeros(len(rows) + 1); c[-1] = -1
    res = linprog(c, A_ub=np.hstack([-G.T, np.ones((len(cols), 1))]), b_ub=np.zeros(len(cols)),
                  A_eq=np.hstack([np.ones((1, len(rows))), [[0]]]), b_eq=[1],
                  bounds=[(0, None)] * len(rows) + [(None, None)], method="highs")
    return res.x[-1]


def test_matches_full_5x5_linear_programme():
    """The 3x3 shortcut must agree with solving the full 5x5 game by simplex."""
    sol = solve(12, 4)
    for n in range(1, 13):
        for r1 in range(5):
            for r2 in range(5):
                assert sol.value(n, r1, r2) == pytest.approx(full_game_value(sol.V[n - 1], r1, r2), abs=1e-7)


def test_optimal_strategy_cannot_be_exploited():
    """Exact check: the best possible reply to the solved strategy gains nothing."""
    rounds, rockets = 40, 6
    sol = solve(rounds, rockets)
    E = np.zeros((rockets + 1, rockets + 1))          # optimal player's lead vs a perfect exploiter
    for n in range(1, rounds + 1):
        En = np.zeros_like(E)
        for r1 in range(rockets + 1):
            for r2 in range(rockets + 1):
                p = list(sol.strategy(n, r1, r2).values())
                replies = [m for m in MOVES if m != ROCKET or r2 > 0]
                En[r1, r2] = min(sum(p[a] * (PAYOFF[a][b] + E[r1 - (a == ROCKET), r2 - (b == ROCKET)])
                                     for a in MOVES if p[a] > 0) for b in replies)
        E = En
    assert np.allclose(E, sol.V[rounds], atol=1e-4)
    assert E[rockets, rockets] == pytest.approx(0, abs=1e-4)


def test_bots_play_legal_matches():
    sol = solve(50, 5)
    bots = default_bots(sol)
    for a in bots:
        for b in bots:
            res = play_match(a, b, 50, 5, seed=1, keep_history=True)
            assert res.illegal_a == res.illegal_b == 0
            assert res.history_a.count(ROCKET) <= 5 and res.history_b.count(ROCKET) <= 5
            assert res.rockets_left_a == 5 - res.history_a.count(ROCKET)


def test_optimal_breaks_even_against_itself():
    sol = solve(50, 5)
    a, b = OptimalBot(sol), OptimalBot(sol)
    diffs = [play_match(a, b, 50, 5, seed=s).diff for s in range(400)]
    m = sum(diffs) / len(diffs)
    se = math.sqrt(sum((d - m) ** 2 for d in diffs) / len(diffs) / len(diffs))
    assert abs(m) < 4 * se + 0.2
