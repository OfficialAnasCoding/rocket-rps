"""Rules of Rocket RPS and the match engine.

Moves
-----
ROCK, PAPER, SCISSORS : normal rules, the winner scores 1 point.
ROCKET : beats rock, paper and scissors for 3 points. Limited supply.
WATER  : beats a rocket for 2 points. Loses to rock, paper and scissors
         (they score 1 point). Unlimited.
Two identical moves score nothing. A rocket is used up whenever it is played.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

ROCK, PAPER, SCISSORS, ROCKET, WATER = range(5)
MOVES = (ROCK, PAPER, SCISSORS, ROCKET, WATER)
PLAIN = (ROCK, PAPER, SCISSORS)
NAMES = ("rock", "paper", "scissors", "rocket", "water")

# What each plain move beats / loses to.
BEATS = {ROCK: SCISSORS, SCISSORS: PAPER, PAPER: ROCK}
LOSES_TO = {v: k for k, v in BEATS.items()}

ROCKET_POINTS = 3
WATER_POINTS = 2
PLAIN_POINTS = 1


def points(a: int, b: int) -> tuple[int, int]:
    """Points scored by (player playing a, player playing b) in one round."""
    if a == b:
        return 0, 0
    if a in PLAIN and b in PLAIN:
        return (PLAIN_POINTS, 0) if BEATS[a] == b else (0, PLAIN_POINTS)
    if a == ROCKET and b in PLAIN:
        return ROCKET_POINTS, 0
    if b == ROCKET and a in PLAIN:
        return 0, ROCKET_POINTS
    if a == WATER and b == ROCKET:
        return WATER_POINTS, 0
    if b == WATER and a == ROCKET:
        return 0, WATER_POINTS
    if a == WATER:  # b is plain
        return 0, PLAIN_POINTS
    return PLAIN_POINTS, 0  # a is plain, b is water


# PAYOFF[a][b] = points for a minus points for b (the zero-sum payoff).
PAYOFF = [[points(a, b)[0] - points(a, b)[1] for b in MOVES] for a in MOVES]


@dataclass
class View:
    """Everything a bot is allowed to see when it chooses a move."""

    rounds_left: int          # including the round about to be played
    total_rounds: int
    my_rockets: int
    opp_rockets: int
    my_score: int
    opp_score: int
    my_history: list[int]     # all my previous moves, oldest first
    opp_history: list[int]    # all the opponent's previous moves

    def legal_moves(self) -> list[int]:
        return [m for m in MOVES if m != ROCKET or self.my_rockets > 0]


@dataclass
class MatchResult:
    name_a: str
    name_b: str
    score_a: int
    score_b: int
    rockets_left_a: int
    rockets_left_b: int
    illegal_a: int = 0
    illegal_b: int = 0
    history_a: list[int] = field(default_factory=list)
    history_b: list[int] = field(default_factory=list)

    @property
    def diff(self) -> int:
        return self.score_a - self.score_b


def play_match(bot_a, bot_b, rounds: int = 1000, rockets: int = 100,
               seed: int | None = None, keep_history: bool = False) -> MatchResult:
    """Play one match. Both bots choose simultaneously each round."""
    rng = random.Random(seed)
    bot_a.reset(rounds, rockets, random.Random(rng.random()))
    bot_b.reset(rounds, rockets, random.Random(rng.random()))
    ra = rb = rockets
    sa = sb = 0
    ha: list[int] = []
    hb: list[int] = []
    illegal_a = illegal_b = 0
    for r in range(rounds, 0, -1):
        a = bot_a.choose(View(r, rounds, ra, rb, sa, sb, ha, hb))
        b = bot_b.choose(View(r, rounds, rb, ra, sb, sa, hb, ha))
        # An illegal move (unknown move, or a rocket with none left) becomes rock.
        if a not in MOVES or (a == ROCKET and ra == 0):
            a, illegal_a = ROCK, illegal_a + 1
        if b not in MOVES or (b == ROCKET and rb == 0):
            b, illegal_b = ROCK, illegal_b + 1
        pa, pb = points(a, b)
        sa += pa
        sb += pb
        ra -= a == ROCKET
        rb -= b == ROCKET
        ha.append(a)
        hb.append(b)
    return MatchResult(bot_a.name, bot_b.name, sa, sb, ra, rb, illegal_a, illegal_b,
                       ha if keep_history else [], hb if keep_history else [])
