# Cost model and ceiling

The procedure of step 3, SpecPilot's Phase 0 ([arXiv 2611.02501](https://arxiv.org/abs/2611.02501),
sections 3.1, 6.1 and 6.4 and appendix B). It runs at the start of every round, at every batch size,
on the round's starting system and the development requests, and its output is the ceiling table.
The batch size B is the number of tasks (or independent requests) the client keeps in flight
(replay.md, rule 4). The engine's batch, the requests one decode step processes, varies between 1 and
B during a run, and every quantity below averages over all the decode steps of a run at B, each step
at its own batch, so that the ceiling and the paired gains describe the same steps.

## Quantities

With N proposed nodes per request, τ(N) is the number of tokens a step commits per request: the
accepted draft tokens plus the one token the target adds after them, summed over the run's decode
steps and divided by the sum of their batches. T_step(N) is the mean time of a decode step of the
run, drafting included, so a request commits τ(N) / T_step(N) tokens per unit time. The ceiling is
the speedup over the starting system that a step would reach if it drafted once, verified the best
tree in one exact pass, and ran every kernel class at its byte floor:

    ceiling = (τ_best / τ) × (T_prof / T_floor) × (T_step / T_prof)

The three ratios are the τ factor, the kernel factor and the overhead factor.

## Procedure

1. **τ and T_step.** Read τ and T_step of the starting system over every decode step of its run at
   B, from the run's scheduler trace. This run, the profiled run of step 5 and the budgets' runs of
   step 3 run one after another on one slot, with nothing else on its machine:

   ```bash
   python skills/optimize-speculative-decoding/scripts/decode_steps.py <run>/trace \
     --start <t_start> --end <t_end>      # the window from <run>/run.json
   ```

   It leaves out only the steps next to a prefill or an idle wait, whose time holds other work, and
   also prints the run's mean batch and its steps per batch.

2. **Lattices.** Record the drafter's lattices in a run of the starting system on development
   requests. A lattice holds, for one drafting step of one request (keyed by the `rid` the client
   sends), the drafter's top candidates at each drafted position with their scores, the score row
   given the committed anchor, and, when the drafter scores pairs, the score of each candidate given
   each candidate at the position before (`exact_replay.py --help` gives the format). The recorder is
   an engine hook, off by default; when the engine has none, adding it is the round's first patch.
3. **T_step(N).** For each node budget N to replay, measure the full step time with N proposed nodes
   per request (drafting, tree construction, verify and commit): the mean step time of `decode_steps.py`
   over a run of chains or trees of N nodes on the same engine, at B, on the same requests and on step
   1's slot, never the verify forward alone. A budget of N takes N + 1 rows per request.
4. **τ factor.** Replay the lattices:

   ```bash
   python skills/optimize-speculative-decoding/scripts/exact_replay.py <lattices.jsonl> \
     --outputs <run>/outputs.jsonl --budgets 8,16,24,32,48,64 --costs <N:T_step(N) in ms,...>
   ```

   The script builds, for every budget, the best-first tree: a node's path score is the sum of its
   edges' scores, an edge's score is the log softmax over its parent's children (of the pair scores
   given the parent when the drafter has them, else of the position's own scores), and the tree of
   budget N is the first N nodes in order of path score. It reports τ(N) of these trees, the chain's
   τ and the oracle's τ, the tokens accepted while the target's token is among the position's
   candidates. Under greedy decoding a tree accepts the longest root path whose tokens equal the
   target's, plus one token; under sampling it uses the verify that samples the target's token at each
   node and continues through the child that holds it. τ_best is τ(N) at the N with the largest
   τ(N) / T_step(N), and the τ factor is τ_best / τ.
5. **T_prof.** On step 1's slot, run the starting system again at B and profile it in windows spread
   through the run (for example a quarter, half and three quarters of the way), and pass one GPU's
   traces of all the windows to
   `scripts/profile_steps.py`, cut at the first kernel of the exact pass with `--anchor-kernel`. T_prof
   is the mean GPU busy time per step over the pooled steps, by kernel class.
6. **T_floor.** T_floor is the mean over the profiled steps of each step's floor at its own batch:
   `decode_steps.py --windows` over the profile windows gives their steps per batch and their context,
   which `byte_floor.py --batch-mix` and `--context-tokens` take. A step's floor sums, class by class,
   the larger of its FLOPs at 60 % of the GPU's peak dense throughput and its bytes at 90 % of peak
   bandwidth. The target GEMMs read every weight once per step, or in a mixture of experts the experts
   the rows reach. Attention reads every request's KV cache once, 1/min(G, KV heads) of it per GPU with
   tensor parallelism over G GPUs; a sliding-window layer reads at most its window. Recurrent layers
   read and write every request's state and update it for every row. The drafter does its own work as
   many times as it runs per step, with its own experts and its own tied output layer. Prefill
   processes the step's prompt tokens with their attention, with one more weight read per prefill
   forward of its own, of the experts its chunk's tokens reach. Sampling reads the logits once. A
   one-shot all-reduce of M bytes moves (G − 1) × M per GPU and a two-shot one 2 (G − 1) / G × M, over
   the link bandwidth. Every other class follows the same rule.
   `scripts/byte_floor.py` computes the weight, attention, recurrent-state, drafter and prefill
   classes from the checkpoint; tensors copied to every GPU, weights shared with the target and
   formats converted at load are corrected by hand. The kernel factor is T_prof / T_floor.
7. **Overhead factor.** T_step / T_prof, with T_step from step 1.
8. **Table.** Write, per batch size: the mean batch, τ, the oracle's τ, τ_best and its N, T_step(N) at
   that N, T_step, T_prof, T_floor by kernel class, the three factors, their product, and the largest
   factor, which is the round's gap.

A tree chosen from the replay is used in a direction only after it is implemented and its live tokens
per step equal the replay's on the same requests within the run's noise.
