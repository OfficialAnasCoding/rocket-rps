# The bots and their strategies

Every bot sees the same information each round: how many rounds are left, how
many rockets each player has, both scores, and every move played so far. They
differ in what they do with it.

The bots fall into three groups:

- **Solved:** Optimal
- **Fixed:** Random, PlainRPS, RocketsEarly, RocketsLate, SpreadRockets, OneRoundNash, BeatLast
- **Learning:** Pattern, Adaptive

"Plain move" below means rock, paper or scissors chosen at random with equal
chance.

---

## Optimal

**Idea:** play the mathematically solved strategy (the Nash equilibrium).

**How it works:** before the tournament, the solver works out the best mix of
moves for every possible situation: rounds left (1 to 1000) × my rockets
(0 to 100) × opponent's rockets (0 to 100), about 10 million situations. Each
round, the bot looks up the current situation and picks a move at random using
those probabilities.

**What it does in practice:**
- Fires a rocket 1/6 of the time whenever it has one.
- Plays water more often the more dangerous the opponent's rockets are: almost
  never at the start of a match, up to half the time when the opponent has
  many rockets and few rounds left, and never once the opponent has run out.
- Splits the rest equally between rock, paper and scissors.

**Strength:** it cannot be beaten on average, even by an opponent who knows
exactly how it works.

**Weakness:** it does not punish mistakes unless they are outright blunders
(such as playing water against a player with no rockets). Against most bots it
finishes about level.

---

## Adaptive

**Idea:** be safe until you understand the opponent, then exploit them, and go
back to safe if it stops working.

**How it works:**
1. **Warm-up (first 30 rounds):** plays exactly like Optimal while it watches.
2. **Prediction:** it keeps a running count of what the opponent played after
   each recent situation (the same predictor as Pattern, described below) and
   gets a prediction with a confidence between 0 and 1.
3. **Exploit:** with probability equal to its confidence, it plays the best
   reply to the prediction. Otherwise it plays the Optimal move. So the more
   sure it is, the more it exploits.
4. **Planning ahead:** when choosing the best reply, it scores each move as
   *points this round + the solver's value of the situation afterwards*. This
   stops it wasting rockets: it only fires when the gain now is worth more than
   keeping the rocket.
5. **Stop-loss:** it tracks how many points its exploiting moves have won or
   lost. If they fall 12 points behind, it assumes the opponent is out-guessing
   it, plays pure Optimal for 100 rounds, then tries again.

**Strength:** wins big against predictable bots and loses very little against
unpredictable ones.

**Weakness:** it pays a small price for testing opponents that cannot be
exploited, so it tends to lose a few points to Optimal.

---

## Pattern

**Idea:** find patterns in the opponent's moves and play the move that scores
best *this round*.

**How it works:** it predicts the opponent's next move by asking "what did
they do the last few times we were in this position?" It looks at three levels
and uses the most detailed one that has enough data:
1. What they played after the same *pair* of previous moves (their last move
   and my last move).
2. What they played after their own last move.
3. What they play overall.

Old observations slowly fade (each round the counts shrink by 3%), so it
follows an opponent who changes behaviour. It then plays whichever legal move
has the highest expected score for this round against the prediction.

**Strength:** destroys anything with a simple pattern. Against BeatLast it wins
almost every round.

**Weaknesses:**
- **It is greedy.** It never thinks about saving rockets, so it fires them
  whenever a rocket looks best this round.
- **It is predictable itself.** Its move is fully determined by its prediction,
  so a smarter bot (Adaptive) can predict the predictor and beat it.

---

## OneRoundNash

**Idea:** solve one round of the game and play that answer every round.

**How it works:** with unlimited rockets, the best mix for a single round is
rocket 1/6, water 1/2, and rock, paper and scissors 1/9 each. This bot plays
that mix every round, adjusting only for what is possible: no rockets when it
has none left, and no water when the opponent has none left.

**Strength:** simple, random, and close to Optimal late in a match.

**Weakness:** it ignores the future. It plays water half the time from round 1,
when the opponent's 100 rockets are spread over 1000 rounds and rarely appear.
Water loses to rock, paper and scissors, so plain bots beat it. This bot shows
why the full solution has to look ahead.

---

## SpreadRockets

**Idea:** a sensible human rule of thumb: pace your rockets, and defend in
proportion to the threat.

**How it works:**
- Fires a rocket with probability *my rockets ÷ rounds left*, which spreads
  them evenly over the rest of the match and uses them all by the end.
- Plays water with probability *opponent's rockets ÷ rounds left* (capped at
  1/2).
- Otherwise plays a plain move.

**Strength:** it gets the main ideas of the solved strategy (random timing,
water that follows the threat) without any computation, and it does well.

**Weakness:** the proportions are a guess. Optimal fires faster (1/6 per round)
and tunes water exactly.

---

## RocketsEarly

**Idea:** rockets score 3, so use them straight away.

**How it works:** fires a rocket every round for the first 100 rounds, then
plays plain moves for the other 900.

**Strength:** beats bots that never play water, such as PlainRPS, by a wide
margin.

**Weakness:** totally predictable. Any bot that notices plays water for 100
rounds and wins 2 points each time.

---

## RocketsLate

**Idea:** save the rockets as a surprise for the end.

**How it works:** plays plain moves until the number of rounds left equals its
number of rockets, then fires a rocket every round to the finish.

**Strength:** same as RocketsEarly: it punishes bots that never defend.

**Weakness:** the same predictability, just at the other end of the match. An
opponent can see it still holds 100 rockets with 100 rounds left, and play
water.

---

## BeatLast

**Idea:** assume the opponent will repeat their last move, and play what beats
it.

**How it works:** after rock it plays paper, after paper scissors, after
scissors rock, after a rocket water, and after water a plain move.

**Strength:** beats opponents who really do repeat themselves. It does well
against RocketsEarly and RocketsLate during their rocket streaks.

**Weakness:** it is a fixed rule with no randomness, so a learning bot can work
out its next move exactly. It is the biggest loser against Pattern and
Adaptive.

---

## PlainRPS

**Idea:** ignore the new moves and play ordinary rock, paper, scissors.

**How it works:** a random plain move every round. It never fires a rocket and
never plays water.

**Strength:** it cannot be out-guessed at rock, paper, scissors, and it beats
bots that play too much water.

**Weakness:** it wastes its 100 rockets (up to 300 points) and has no defence
against the opponent's.

---

## Random

**Idea:** no strategy at all. This is the baseline.

**How it works:** picks any legal move with equal chance: 1/5 each while it has
rockets, 1/4 each after that.

**Strength:** unpredictable, and it does use its rockets.

**Weakness:** it plays far too much water, including when the opponent has no
rockets left. It is the one bot that Optimal beats clearly, because these are
true blunders and not just habits.
