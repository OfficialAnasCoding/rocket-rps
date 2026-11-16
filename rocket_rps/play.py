"""Play against a bot in the terminal, or ask the solver about one situation.

    python -m rocket_rps.play --bot Optimal --rounds 30 --rockets 3
    python -m rocket_rps.play --lookup 300 70 50
"""
from __future__ import annotations

import argparse
import random

from .bots import default_bots
from .game import NAMES, ROCKET, View, points
from .solver import load_or_solve

KEYS = {"r": 0, "p": 1, "s": 2, "k": 3, "w": 4}


def lookup(solution, n, r1, r2):
    print(f"\n{n} rounds left, you have {r1} rockets, opponent has {r2}:")
    for name, mine, theirs in zip(NAMES, solution.strategy(n, r1, r2).values(),
                                  solution.strategy(n, r2, r1).values()):
        print(f"  {name:<9} you {mine:6.1%}   opponent {theirs:6.1%}")
    print(f"  expected final lead for you: {solution.value(n, r1, r2):+.2f} points")


def play(bot, rounds, rockets):
    rng = random.Random()
    bot.reset(rounds, rockets, rng)
    mine = theirs = rockets
    my_score = bot_score = 0
    my_hist, bot_hist = [], []
    print(f"You are playing {bot.name}: {bot.description}")
    print("Moves: r = rock, p = paper, s = scissors, k = rocket, w = water\n")
    for n in range(rounds, 0, -1):
        b = bot.choose(View(n, rounds, theirs, mine, bot_score, my_score, bot_hist, my_hist))
        while True:
            key = input(f"[{n} left | you {my_score} ({mine} rockets) | bot {bot_score} ({theirs} rockets)] > ").strip().lower()
            if key in KEYS and (KEYS[key] != ROCKET or mine > 0):
                break
            print("  Type r, p, s, k or w (k only while you have rockets).")
        a = KEYS[key]
        pa, pb = points(a, b)
        my_score, bot_score = my_score + pa, bot_score + pb
        mine -= a == ROCKET
        theirs -= b == ROCKET
        my_hist.append(a)
        bot_hist.append(b)
        print(f"  you: {NAMES[a]}, bot: {NAMES[b]}  ->  you +{pa}, bot +{pb}")
    print(f"\nFinal score: you {my_score}, {bot.name} {bot_score}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Play Rocket RPS against a bot")
    ap.add_argument("--bot", default="Optimal")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--rockets", type=int, default=3)
    ap.add_argument("--lookup", type=int, nargs=3, metavar=("ROUNDS_LEFT", "MY_ROCKETS", "THEIR_ROCKETS"),
                    help="print the optimal strategy for one situation in the full game and exit")
    args = ap.parse_args(argv)
    if args.lookup:
        n, r1, r2 = args.lookup
        lookup(load_or_solve(max(n, 1), max(r1, r2)), n, r1, r2)
        return
    solution = load_or_solve(args.rounds, args.rockets, cache_dir=None, verbose=False)
    bots = {b.name.lower(): b for b in default_bots(solution)}
    if args.bot.lower() not in bots:
        raise SystemExit(f"Unknown bot. Choose from: {', '.join(b.name for b in bots.values())}")
    play(bots[args.bot.lower()], args.rounds, args.rockets)


if __name__ == "__main__":
    main()
