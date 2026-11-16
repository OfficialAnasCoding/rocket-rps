"""The bots. Each one is a different idea about how to play Rocket RPS.

To write your own, subclass Bot and implement choose(view). The view tells
you the rounds left, both rocket counts, both scores and both move histories
(see game.View). Return one of ROCK, PAPER, SCISSORS, ROCKET, WATER.
"""
from __future__ import annotations

import random

from .game import (BEATS, LOSES_TO, MOVES, PAPER, PAYOFF, PLAIN, ROCK, ROCKET,
                   SCISSORS, WATER, View)
from .solver import Solution


class Bot:
    name = "Bot"
    description = ""

    def reset(self, rounds: int, rockets: int, rng: random.Random) -> None:
        """Called at the start of every match."""
        self.rng = rng

    def choose(self, view: View) -> int:
        raise NotImplementedError

    def plain(self) -> int:
        """A uniformly random rock, paper or scissors."""
        return self.rng.choice(PLAIN)


# --------------------------------------------------------------------------
# The solved strategy
# --------------------------------------------------------------------------

class OptimalBot(Bot):
    name = "Optimal"
    description = "Plays the exact Nash equilibrium from the solver. Cannot be beaten on average."

    def __init__(self, solution: Solution):
        self.solution = solution

    def reset(self, rounds, rockets, rng):
        super().reset(rounds, rockets, rng)
        if rounds > self.solution.rounds or rockets > self.solution.rockets:
            raise ValueError("The solution does not cover a match this large.")

    def choose(self, view):
        return sample_equilibrium(self.solution, view, self.rng)


def sample_equilibrium(solution: Solution, view: View, rng: random.Random) -> int:
    pk, pw = solution.policy[view.rounds_left, view.my_rockets, view.opp_rockets]
    u = rng.random()
    if u < pk:
        return ROCKET
    if u < pk + pw:
        return WATER
    return rng.choice(PLAIN)


# --------------------------------------------------------------------------
# Simple fixed strategies
# --------------------------------------------------------------------------

class RandomBot(Bot):
    name = "Random"
    description = "Picks uniformly at random from all the moves it is allowed to play."

    def choose(self, view):
        return self.rng.choice(view.legal_moves())


class PlainBot(Bot):
    name = "PlainRPS"
    description = "Ignores the special moves and plays random rock, paper, scissors."

    def choose(self, view):
        return self.plain()


class RocketsEarlyBot(Bot):
    name = "RocketsEarly"
    description = "Fires a rocket every round until it runs out, then plays random rock, paper, scissors."

    def choose(self, view):
        return ROCKET if view.my_rockets > 0 else self.plain()


class RocketsLateBot(Bot):
    name = "RocketsLate"
    description = "Plays random rock, paper, scissors, saving every rocket for the final rounds."

    def choose(self, view):
        if view.my_rockets > 0 and view.rounds_left <= view.my_rockets:
            return ROCKET
        return self.plain()


class SpreadRocketsBot(Bot):
    name = "SpreadRockets"
    description = ("Spreads its rockets evenly at random over the rest of the match and plays "
                   "water in proportion to the opponent's remaining rockets.")

    def choose(self, view):
        n = view.rounds_left
        u = self.rng.random()
        p_rocket = view.my_rockets / n
        p_water = min(0.5, view.opp_rockets / n)
        if u < p_rocket:
            return ROCKET
        if u < p_rocket + p_water * (1 - p_rocket):
            return WATER
        return self.plain()


class OneRoundBot(Bot):
    name = "OneRoundNash"
    description = ("Plays the one-round equilibrium every round (rocket 1/6, water 1/2, others 1/9 "
                   "each) without thinking about the future.")

    def choose(self, view):
        u = self.rng.random()
        p_rocket = 1 / 6 if view.my_rockets > 0 else 0.0
        p_water = 1 / 2 if view.opp_rockets > 0 else 0.0   # water is pointless against no rockets
        if u < p_rocket:
            return ROCKET
        if u < p_rocket + p_water:
            return WATER
        return self.plain()


class BeatLastBot(Bot):
    name = "BeatLast"
    description = "Plays whatever would have beaten the opponent's previous move."

    def choose(self, view):
        if not view.opp_history:
            return self.plain()
        last = view.opp_history[-1]
        if last == ROCKET:
            return WATER
        if last == WATER:
            return self.plain()
        return LOSES_TO[last]


# --------------------------------------------------------------------------
# Bots that learn about their opponent
# --------------------------------------------------------------------------

class _Predictor:
    """Predicts the opponent's next move from their history.

    Keeps counts of what the opponent played after each context, where the
    context is (their last move, my last move), then just (their last move),
    then nothing. The deepest context with enough data is used. Old
    observations fade so that the prediction follows a changing opponent.
    """

    def __init__(self, decay: float = 0.97, min_weight: float = 4.0):
        self.decay = decay
        self.min_weight = min_weight
        self.tables: list[dict] = [{}, {}, {}]
        self.seen = 0

    def _contexts(self, my_hist, opp_hist, upto):
        if upto >= 1:
            yield 0, (opp_hist[upto - 1], my_hist[upto - 1])
            yield 1, opp_hist[upto - 1]
        yield 2, None

    def update(self, my_hist, opp_hist):
        while self.seen < len(opp_hist):
            t = self.seen
            for level, ctx in self._contexts(my_hist, opp_hist, t):
                row = self.tables[level].setdefault(ctx, [0.0] * 5)
                for m in MOVES:
                    row[m] *= self.decay
                row[opp_hist[t]] += 1.0
            self.seen += 1

    def predict(self, my_hist, opp_hist, opp_rockets) -> tuple[list[float], float]:
        """Return (probabilities for the opponent's next move, confidence 0..1)."""
        probs, weight = [0.2] * 5, 0.0
        for level, ctx in self._contexts(my_hist, opp_hist, len(opp_hist)):
            row = self.tables[level].get(ctx)
            if row and sum(row) >= self.min_weight:
                weight = sum(row)
                probs = [(c + 0.2) / (weight + 1.0) for c in row]
                break
        if opp_rockets == 0:   # they cannot fire
            probs[ROCKET] = 0.0
            total = sum(probs)
            probs = [p / total for p in probs]
        return probs, min(1.0, weight / 20.0)


class PatternBot(Bot):
    name = "Pattern"
    description = ("Looks for patterns in the opponent's moves and plays the move that scores "
                   "best this round against its prediction. Greedy: it does not plan ahead.")

    def reset(self, rounds, rockets, rng):
        super().reset(rounds, rockets, rng)
        self.predictor = _Predictor()

    def choose(self, view):
        self.predictor.update(view.my_history, view.opp_history)
        probs, _ = self.predictor.predict(view.my_history, view.opp_history, view.opp_rockets)
        best, best_ev = [], -1e9
        for m in view.legal_moves():
            ev = sum(probs[o] * PAYOFF[m][o] for o in MOVES)
            if ev > best_ev + 1e-9:
                best, best_ev = [m], ev
            elif abs(ev - best_ev) <= 1e-9:
                best.append(m)
        return self.rng.choice(best)


class AdaptiveBot(Bot):
    name = "Adaptive"
    description = ("Starts on the solved strategy, learns the opponent's habits, then exploits "
                   "them using the solver's values to plan ahead. Falls back to the solved "
                   "strategy if exploiting starts losing.")

    def __init__(self, solution: Solution, warmup: int = 30, stop_loss: float = 12.0):
        self.solution = solution
        self.warmup = warmup
        self.stop_loss = stop_loss

    def reset(self, rounds, rockets, rng):
        super().reset(rounds, rockets, rng)
        self.predictor = _Predictor()
        self.exploit_gain = 0.0      # points won minus lost on rounds where it exploited
        self.exploited_last = False
        self.resume_at = None        # after a stop-loss: rounds_left at which to try again

    def choose(self, view):
        V = self.solution.V
        n, r1, r2 = view.rounds_left, view.my_rockets, view.opp_rockets
        if self.exploited_last:
            self.exploit_gain += PAYOFF[view.my_history[-1]][view.opp_history[-1]]
            if self.exploit_gain < -self.stop_loss:
                # The opponent is out-guessing us. Play safe for 100 rounds, then try again.
                self.resume_at = n - 100
                self.exploit_gain = 0.0
        self.predictor.update(view.my_history, view.opp_history)
        probs, confidence = self.predictor.predict(view.my_history, view.opp_history, r2)

        if self.resume_at is not None and n <= self.resume_at:
            self.resume_at = None
        warmed_up = view.total_rounds - n >= self.warmup
        if not warmed_up or self.resume_at is not None or self.rng.random() > confidence:
            self.exploited_last = False
            return sample_equilibrium(self.solution, view, self.rng)

        # Best response to the prediction, counting what the future is worth.
        best, best_ev = [], -1e18
        for m in view.legal_moves():
            ev = 0.0
            for o in MOVES:
                if probs[o] == 0.0:
                    continue
                ev += probs[o] * (PAYOFF[m][o] + V[n - 1, r1 - (m == ROCKET), r2 - (o == ROCKET)])
            if ev > best_ev + 1e-9:
                best, best_ev = [m], ev
            elif abs(ev - best_ev) <= 1e-9:
                best.append(m)
        self.exploited_last = True
        return self.rng.choice(best)


def default_bots(solution: Solution) -> list[Bot]:
    """The line-up used by the tournament."""
    return [
        OptimalBot(solution),
        AdaptiveBot(solution),
        PatternBot(),
        OneRoundBot(),
        SpreadRocketsBot(),
        RocketsEarlyBot(),
        RocketsLateBot(),
        BeatLastBot(),
        PlainBot(),
        RandomBot(),
    ]
