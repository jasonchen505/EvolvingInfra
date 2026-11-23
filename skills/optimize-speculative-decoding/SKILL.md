---
name: optimize-speculative-decoding
description: Run SpecPilot's loop (arXiv 2611.02501) on one speculative-decoding deployment - compute the ceiling, explore directions in parallel, promote changes on paired runs, check that every change stays lossless, and seal each round on unseen requests. Use when asked to make speculative decoding faster for a target, drafter and GPU, or to decide whether a change speeds it up without changing outputs. Not for running Evolving Infra's replay benchmark; use `run-benchmark`.
---

# Optimize speculative decoding

This skill makes one deployment's speculative decoding faster without changing what the target model
outputs. A deployment is a serving regime, a drafter, a target model and a GPU platform; the target
and its drafter run on `decoupled-engine`. The steps below are SpecPilot's workflow
([arXiv 2611.02501](https://arxiv.org/abs/2611.02501)) and are followed in order. Three references
hold the procedures the steps call: [references/cost-model.md](references/cost-model.md) computes the
ceiling, [references/lossless.md](references/lossless.md) holds the lossless checks with their pass
criteria, and [references/replay.md](references/replay.md) the rules every run follows.

## Scope

- **Reads**: `evolving_infra/engine/`: the speculative workers, kernels and server arguments;
  `skills/optimize-speculative-decoding/references/`: the procedures.
- **Writes**: `evolving_infra/engine/`: patches behind options whose defaults reproduce stock;
  `tests/engine/`: their tests; `docs/`: the options they add. Notes, measurements and memory entries
  go to a work directory outside the repository.
- **Never changes**: `skills/optimize-speculative-decoding/scripts/` and the frozen configuration of
  step 2 during a study: they are the controller, which the optimizing agents call and never modify. A
  defect found in one goes to the owner, who fixes it in its own commit with a test; every
  measurement it touched is redone.
- **Outside the repository**: GPU jobs and sub-agents, agreed with the owner before the first job.

## Inputs

The owner gives five inputs, fixed for the study. The deployment: the target and the drafter with
their revisions and precision, the tokenizer, processor and chat-template settings, the engine build,
the GPU, the GPUs per engine, and the batch sizes: a batch size B is the number of tasks, or of
independent requests, the client keeps in flight (replay.md, rule 4). Stock: the drafter's released
configuration on the engine. The requests: real traffic in replay.md's format, with a group field for
the gain (the paper's benchmarks). The decoding rule: greedy, or sampling with its temperature, top-k
and top-p. The budget: GPUs at once, time, rounds, and whether sub-agents may run.

## Steps

Steps 1 and 2 run once; each round then runs steps 3 to 8. Every step ends with a gate, its Check,
and the next step starts only when the gate holds. The next round starts from the round's final system
if it passed step 7, else from the round's starting system.

The main session is the controller and the master. The controller alone holds the sealed requests,
the frozen configuration and the raw results; it runs every comparison and lossless check and gives
sub-agents only their verdicts and the development requests. The master plans directions and writes
no patch. A sub-agent implements one direction in its own git worktree and submits it as a commit; it
never reads the sealed requests or changes the controller. Nothing is derived from the sealed
requests before step 7, and each round's step 7 uses its own sealed part once. The controller notes
every number with its command, every decision with its reason, and every direction with its factor,
headroom and outcome.

Work runs in parallel wherever the rules below allow. A run's times (step times, request rates,
profiles) depend on what else its machine runs; its outputs (tokens, log-probabilities, scores,
lattices, test counts) do not.

Runs in parallel:
- Runs used only for their outputs, on any free slots: the lossless procedures' runs and their
  scoring, the AR runs of Procedure 1, Procedure 4's tests, the lattice recording of step 3.
- Measurements that are not compared with each other, on separate machines: the batch sizes of steps
  2 and 3, and comparisons of different patches against the same current best.
- The sub-agents of steps 4 and 6, each on its own slots, also while the controller measures.
- Sub-steps 2 to 5 of step 2: each waits only for the runs it reads. If stock fails Procedure 1,
  everything sub-steps 3 to 5 measured is measured again on the fixed stock (Procedure 5).

Never in parallel:
- Anything else on a machine while it measures times.
- The pairs of one comparison, the stock-against-stock pairs of the noise included: one after another
  on the same two slots of one machine, with the slots swapped between pairs. The two runs of one pair
  run at the same time; that is what makes them a pair.
- At one batch size, the cost model's timing runs: the starting system's run (in round 1, the run of
  step 2.3), its profiled run and the runs of the replayed budgets, one after another on one slot
  (cost-model.md, steps 1, 3 and 5).
- Decisions: one at a time. When a patch is promoted, every undecided patch is timed again against the
  new current best.

### 1. Split the requests

Split the requests into the development file and one sealed part per planned round (replay.md,
rule 6):

```bash
python skills/optimize-speculative-decoding/scripts/split_requests.py <requests.jsonl> \
  --out-dir <work>/requests --salt <study name> --sealed-parts <rounds> [--group-field <task field>]
```

Every run from here on follows replay.md, with the development cap on the output length (the paper:
2048 tokens).

**Check**: the counts and sha256 of every file, the parts and the cap are in the notes.

### 2. Tune stock, check it, measure the starting point, freeze the controller

1. Tune stock for this GPU and regime with the fastest upstream engine options that leave outputs
   unchanged; on H100 the paper's stock is the released configuration.
2. Run lossless.md Procedure 1 on stock. If stock fails, follow Procedure 5 and continue only with the
   fixed stock.
3. At every batch size, run stock on the development requests and read τ, T_step and the context
   over all the run's decode steps with `decode_steps.py` (cost-model.md, procedure step 1). The
   request set must hold at least B tasks, with `--session-copies` when needed: the client refuses a
   smaller one, which could not keep B in flight.
4. Under greedy decoding, run stock against stock at batch size 1, when the study has it, at least
   four pairs, and freeze the noise, the root mean square of (gain − 1): `paired_ab.py --noise-only
   --base stock <run dirs> --expect-batches 1 --groups <requests.jsonl> --group-field <group field>`.
   Under sampling, measure no noise. The noise is used only where the two systems' outputs are
   identical on every request (step 5); when a comparison at another batch size has identical
   outputs, `paired_ab.py` asks for the noise of that batch size, measured the same way.
5. Under greedy decoding, calibrate δ (lossless.md, Procedure 2, step 1).
6. Write the frozen configuration: the deployment, stock, the request files and their sha256, the batch
   sizes, the slots, the decoding rule, the caps, the noise, δ, K and the scorer's settings. It changes
   only by the owner's decision, recorded in the notes.

**Check**: stock passed Procedure 1; τ, T_step and the context per batch size, and the noise where
step 2.4 measured it, are in the notes; the frozen configuration exists.

### 3. Phase 0: compute the ceiling

Follow the procedure of cost-model.md at every batch size, on the round's starting system:

    ceiling = (τ_best / τ) × (T_prof / T_floor) × (T_step / T_prof)

The largest of the three factors is the round's gap. The ceiling assumes one draft per exact pass.

**Check**: the ceiling table of cost-model.md, step 8, exists for every batch size.

### 4. Phase 1: explore directions in parallel

1. Plan directions only after step 3's table exists. A direction is an idea aimed at one of the
   largest factors; it names that factor and its headroom, the expected gain. Work outside the decode
   step, such as tokenization, image processing or scheduling between requests, is not a direction.
2. Run the most promising few directions at once, one sub-agent each. A sub-agent implements its
   direction as patches behind options whose defaults reproduce stock.
3. The sub-agent checks the headroom with the cheapest evidence first: byte arithmetic, exact replay,
   a microbenchmark, then a timing run on one GPU. It builds a new algorithm only when the headroom,
   net of runtime overhead, is at least 3 %, and keeps smaller measured patches for step 6. It closes a
   direction only after measuring the implemented mechanism, unless that takes more than 10
   GPU-minutes.
4. Before the controller times a patch, the patch passes its correctness tests and, on development
   requests at the largest batch size, lossless.md Procedure 2 under greedy decoding, or Procedure 3
   under sampling and Procedure 4 when it changes the sampling code. A patch that fails goes back to
   its sub-agent. A patch kept for step 6 is checked as part of the stack that is timed.

**Check**: every direction has its factor, headroom, evidence, outcome and decision in the notes, and
every patch sent to step 5 passed its lossless procedures.

### 5. Promote with paired runs

1. Time the patch against the current best: stock in the first round, then the last promoted system.
   A pair is one run of each system at the same time on the two slots of one machine, on the same
   development requests, at the same batch size, to the end.
2. Fix the number of pairs before the first run.
3. Run the comparison once:

   ```bash
   python skills/optimize-speculative-decoding/scripts/paired_ab.py --base <current best> --cand <patch> \
     <run dirs ...> [--noise <frozen B:X,...>] --expect-pairs <n> --expect-batches <B,...> \
     --groups <requests.jsonl> --group-field <group field>
   ```

The gain of a pair is the geometric mean of the per-request output-rate ratios within each group,
combined by an equal-weight geometric mean over the groups. The patch is promoted at a batch size when
the 95 % lower bound of the gain exceeds 3 % when the outputs of the two systems differ on any
request, else 1 % plus twice the frozen noise of that batch size.

**Check**: the decision, its command and its numbers are in the notes.

### 6. Phases 2 and 3: integrate, tune kernels

An integration sub-agent stacks the patches that gain too little to be promoted alone on the current
best. A kernel sub-agent profiles the current best and tunes the kernels furthest from their byte
floors. Step 4's lossless procedures run once on the system that is timed, the stack, not on each
patch in it; when the stack fails, its patches are checked one at a time to find the one that fails.
The stack is timed under step 5's rule.

**Check**: every promotion of the round has its paired result.

### 7. Phase 4: sealed final

1. On the round's sealed part, time the final system against stock, two paired runs per batch size:

   ```bash
   python skills/optimize-speculative-decoding/scripts/paired_ab.py --sealed --base stock --cand <final> \
     <run dirs ...> --expect-pairs 2 --expect-batches <B,...> --groups <sealed part> --group-field <group field>
   ```

2. Run lossless.md Procedure 2 under greedy decoding, or Procedure 3 under sampling, on the sealed part
   at every batch size, and on long outputs at the largest batch size.
3. The final passes when it is faster than stock at every batch size (`--sealed` exits 0) and every
   lossless procedure passes; otherwise it is rejected whole.

**Check**: the sealed gains per batch size and the lossless verdicts are in the notes.

### 8. Distill memory

Write one memory entry per promoted mechanism in the work directory: its code delta against stock plus
the entries it requires, the commit of the system it was promoted in, its options and variants, its
correctness contract, and a summary of its evidence, limits and untested regimes. Keep an entry only if
applying it with its requirements to stock reproduces that commit's code exactly. A new deployment
starts from stock plus chosen entries, tunes and times them again, and does not count them as new
discoveries.

**Check**: for every entry, stock with the entry and its requirements applied has the same tree as
the commit it was promoted in (`git rev-parse HEAD^{tree}` after committing them equals
`git rev-parse <commit>^{tree}`).

## Returns

The frozen configuration, stock's Procedure 1 verdict, and the GPU-hours and time used. Per round: the
ceiling table, every direction with its factor, headroom, evidence and outcome, and every promotion
with its paired gain. Per sealed final: the gain over stock at each batch size and the lossless
verdicts. Then the memory entries, what was not run, and the owner's decisions on which options to
turn on by default.
