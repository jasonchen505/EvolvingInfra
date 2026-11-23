# Lossless checks

Every system the controller times and every round's final must pass the procedure below that applies
to it. A system that fails is not timed, not promoted and not sealed. The checks compare a system
with stock, so stock passes its own check (Procedure 1) before any other check runs.

| When | Greedy decoding | Sampling |
|---|---|---|
| Step 2, once, on stock | Procedure 1, and Procedure 2 step 1 | Procedure 1, and Procedure 4 on stock's sampling code |
| Step 4, every system before it is timed (a patch, or a stack in step 6) | Procedure 2 at the largest batch size | Procedure 3 at the largest batch size, and Procedure 4 if the system changes the sampling code |
| Step 7, every sealed final | Procedure 2 on the sealed part at every batch size, and on long outputs at the largest | Procedure 3 on the sealed part at every batch size, and on long outputs at the largest |

A system is lossless under greedy decoding when every accepted token lies on a path verified by a pass
over the exact target weights (an exact pass) and equals that pass's argmax, up to floating-point
reduction order at near-ties. It is lossless under sampling when, at every position, its output token
is distributed as the target distribution p: the exact pass's logits after the decoding rule's
temperature, top-k and top-p.

All procedures use one scorer: the stock engine without speculation and with an FP32 output layer
(`--enable-fp32-lm-head`). `scripts/fixed_batch_client.py --score-from <run> --score-top K` feeds a
run's outputs to it teacher-forced, with each turn's prompt rebuilt as the run built it (replay.md),
and records at every output position the scorer's first choice and its K most likely tokens with their
log-probabilities. `scripts/token_forcing.py` gives no verdict (exit 1) for a scorer with speculation
or without the FP32 output layer, a scorer or K that differs between the two sides, a run that did not
finish every request, a request of the run without a complete score, or runs that hold different
requests.

## Procedure 1: stock against AR

1. At the smallest and the largest batch size, run AR (the target without speculation) and stock on
   the development requests, replayed as replay.md describes, at temperature 1 with the deployment's
   top-k and top-p, with `fixed_batch_client.py --logprobs K`.
2. Score both runs with `--score-top K`, the same K. K is large enough when, at every position, the
   scorer's K tokens hold every token the decoding rule can output: K ≥ top_k when top_k is set,
   otherwise tokens holding at least top_p of the probability (0.999 when top_p is 1).
3. Run `token_forcing.py probability --base <AR's scored runs> --cand <stock's scored runs>`.
4. Under sampling, run Procedure 4 on stock's sampling code.

Stock passes when every command exits 0 (the criteria are those of Procedure 3). Stock's scored run
at the largest batch size is the reference run of Procedure 3. If stock fails, follow Procedure 5.

## Procedure 2: token forcing (greedy decoding)

1. Once, after Procedure 1, make stock's reference and calibrate δ. At the largest batch size, run AR
   greedily on the development requests; stock's greedy run there is the one of step 2.3. Score
   stock's run with `--score-top K`, and again with another `--chunked-prefill-size` of the scorer,
   for the audit. Run `token_forcing.py calibrate --ar <AR's run> --stock <stock's run> --scored
   <stock's scored run>`. At each request's first token where stock departs from AR, it takes the
   log-probability gap between the two tokens; δ is the largest gap rounded up to a multiple of 1/8.
   δ goes into the frozen configuration, and stock's two scored runs are the reference of every later
   check on development requests.
2. Run the system greedily at the largest batch size on the same requests, score its run, and score it
   again with the audit's chunk size. For a sealed final, run stock and the final greedily on the
   sealed part at every batch size, and at the largest batch size again with a cap above the
   development cap (the paper: 4096 tokens against 2048); score every run both ways.
3. Run `token_forcing.py check --base <stock's scored runs> --cand <the system's> --delta <δ>
   --rescored <the system's rescored runs> --rescored-base <stock's rescored runs>`, and for the long
   outputs of a sealed final add `--beyond <development cap>`. The rescored runs list one run per
   scored run, in the same order, from a scorer that differs only in `--chunked-prefill-size`.

A mismatch is an output token that differs from the scorer's first choice, and its margin is the
log-probability gap from the first choice to it. The system passes when the command exits 0, which
requires both rules:

- rate rule: the 95 % interval of (the system's mismatch rate − stock's) has its upper end at or below
  0.5 percentage points. An interval whose lower end is above 0.5 points fails; an interval across 0.5
  points is no pass, and more requests are scored;
- event rule: a one-sided binomial test does not find the system's unexplained large-margin events,
  mismatches with a margin above 2δ, more frequent than stock's at p < 0.05. An event is explained
  when stock has the same mismatch at the same request, position, prefix and token, or when the
  rescored runs have another first choice there.

## Procedure 3: probability check (sampling)

1. At the largest batch size, run the system on the same requests as stock's reference run, the same
   way: replayed as replay.md describes, at temperature 1 with the deployment's top-k and top-p, with
   `--logprobs K`. For a sealed final, run stock and the final this way on the sealed part at every
   batch size, and at the largest batch size again with a cap above the development cap (the paper:
   4096 tokens against 2048).
2. Score the system's runs with `--score-top K`, the same K and the same scorer as stock's.
3. Run `token_forcing.py probability --base <stock's scored runs> --cand <the system's scored runs>`,
   and for the long outputs add `--beyond <development cap>`.

At every output position the script computes d, a lower bound of the total variation distance
between the distribution the run's engine computed for that token and the scorer's distribution for
the same prefix. A token missing from one side's list has at most that side's smallest listed
probability, and the missing tokens together at most its unlisted mass, 1 minus its listed
probability. With C the tokens in both lists and A the tokens only the run lists:

    up   = Σ over C of max(0, p_run − p_scorer)
           + max(0, p_run(A) − min(the scorer's unlisted mass, |A| × the scorer's smallest listed probability))
    down = up with the run and the scorer swapped
    d    = max(up, down)

With the same tokens in both lists, d is the distance over those tokens with every other token merged
into one category. The system passes when the command exits 0, which requires, at every batch size
checked, at least 10 000 positions on each side and both rules:

- level rule: at the 50th, 90th, 99th and 99.9th percentiles, the system's d is at most
  max(2 × stock's d at that percentile, 10⁻⁴);
- event rule: the positions where the system's d exceeds max(2 × the largest d of stock, 10⁻⁴) are
  not significant in a one-sided Fisher exact test against stock's count there, which is zero
  (p ≥ 0.05); with equal positions on both sides, at most four such positions.

## Procedure 4: rule check (sampling code)

The sampling code is the code that turns the target's logits and the draft into output tokens: the
temperature, top-k and top-p processing, the acceptance test, residual sampling, the extra token after
a fully accepted draft, the acceptance of a tree or of several candidates, the draft distribution q
handed to them, and the way drafts are drawn.

1. Write a derivation in the correctness contract that, for any target distribution p and draft
   distribution q, the token the code outputs at each position is distributed as p. For the standard
   rule: a token y is output as the accepted draft with probability q(y) × min(1, p(y)/q(y)) =
   min(q(y), p(y)) and by residual sampling with probability max(0, p(y) − q(y)), which sum to p(y).
2. Write a GPU test that runs the engine's own code, on the engine's GPUs, with fixed p and q at
   every position of the chain (every node of a tree), in cases of six kinds: flat (every token
   equally likely), peaked (one token at 0.9 or more), near_tie (the two most likely tokens within
   0.01), draft_excess (q gives probability to tokens p gives 0), target_outside_draft (p gives
   probability to tokens q gives 0), and top_p (p given as logits at the deployment's temperature,
   top-k and top-p). Drafts are drawn from q by the engine's draft code when it changed, else by the
   test. Each tested position is reached by at least 100 000 draws, and in every draw every GPU of the
   engine outputs the same token. The test writes one JSON line per case and position:
   `{"case", "kind", "position", "expected": {token: probability}, "counts": {token: count},
   "ranks_agree": true}`, with expected the p of that position after the decoding rule and counts the
   output tokens of the draws that reached it.
3. Run the same cases on four broken copies of the code, each line marked with `"control"`: `no_q`
   accepts when the uniform draw is below p(x); `wrong_q` uses the q of another position;
   `residual_p` samples the residual from p instead of max(0, p − q); `bias` moves 0.01 of
   probability from one token to another in p.
4. Run `token_forcing.py rules --tests <lines>`.

The script tests each line's counts against its expected probabilities with a chi-square
goodness-of-fit test at level 0.01 / m for m test lines. Tokens with expected counts below 5 are pooled
into cells of expected count at least 5; when they cannot fill a cell beside a single other cell, an
exact binomial test of their total count replaces the chi-square test. A count on a token of
probability 0 fails a line. The code passes when the command exits 0: no test
line fails, and every control fails at least one line. When a control passes, add draws or cases and
run steps 2 to 4 again. The script gives no verdict when a line has fewer than 100 000 draws, its
expected probabilities do not sum to 1, a test line's GPUs disagree, or a kind or a control is
missing.

## Procedure 5: stock fails Procedure 1

A lossy stock is never the reference of a lossless check or the baseline of a gain.

1. Locate the cause: compare d by turn and by position; the first token of a turn comes from the
   prefill, the others from the verify.
2. Fix it as a patch behind an option.
3. Run Procedure 1 on stock with the fix; repeat steps 1 to 3 until it passes.
4. The owner decides that the fixed stock is the study's stock; record it in the notes.
5. Measure again on the fixed stock everything measured on the lossy one: the noise, the starting
   point, the ceiling's inputs and the reference runs.
