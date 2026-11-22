---
name: optimize-speculative-decoding
description: Find lossless speedups of speculative decoding for one model, draft method and GPU with the SpecPilot cost model, and prove each with paired measurements. Use when asked to make decoding faster with speculative decoding, choose draft settings, or decide whether an engine change speeds up decode steps. Not for comparing a whole run with the paper's reference; use `run-benchmark`.
---

# Optimize speculative decoding

Find changes that make speculative decoding faster for one model, draft method and GPU without
changing what the model outputs, each backed by paired measurements. The model and its draft
model must already run on `decoupled-engine`.

## Scope

- **Reads**: `evolving_infra/engine/`: the speculative decoding workers, kernels and server
  arguments; `examples/`: the starting configurations; `skills/optimize-speculative-decoding/references/`:
  the cost model.
- **Writes**: `evolving_infra/engine/`: changes, each behind a flag or environment variable that
  is off by default; `examples/`: a configuration with the promoted settings. Measurements go
  to a work directory outside the repository.
- **Never changes**: the engine's defaults and the tokens it commits: with every new flag off
  the engine behaves as before, and with it on the outputs stay lossless under the decoding
  rule (Inputs). `benchmarks/`: other results are compared with its
  replay settings.
- **Outside the repository**: GPU jobs. A measured run holds the GPUs of one engine for its
  start and a few minutes of decoding; a paired comparison takes two nodes with at least two
  runs of each arm on each. Agree on the GPU budget with the owner before the first job.

## Inputs

- The starting configuration: model, draft method and its settings, as engine arguments or a
  config under `examples/`. Ask the owner when missing; do not guess.
- The GPU type and the number of GPUs per engine, or the owner's rule for choosing it per batch
  size. A rule needs a definition of fitting: for example, the token capacity the engine reports
  at start against a high percentile of the summed context (prompt and output) of B requests drawn
  from the exploration set.
- The batch sizes to optimize: how many requests one engine decodes at once in the target use.
- A request set from the traffic to be served: one JSON object per line with `id`, the prompt
  as `input_ids` or as `text` (with `image_data` for images), and `max_new_tokens`. How many
  draft tokens are accepted depends on the content, so the prompts must be real traffic.
- The decoding rule, and what lossless means under it. Under greedy decoding a committed token
  must be the target's most likely token. Under sampling the owner chooses:
  - **the same draws**: every committed token is the token the target draws itself with the same
    key (a seed per request and the position). The engine without speculation must draw with that
    key, and so must the verify step of the draft method. `fixed_batch_client.py --keyed-seed`
    checks the engine before every run; step 7 checks the outputs.
  - **the same distribution**: committed tokens follow the target's distribution (rejection
    sampling). Step 7's position-by-position check does not apply; agree on the check with the
    owner.
- The GPU budget and the time limit. Keep enough of both for step 7.
- Never ask for a token or password; the person logs in and gives a path to the credentials file.

## Steps

The method is SpecPilot's; [references/cost-model.md](references/cost-model.md) defines every
quantity below and how it is measured. The speedup of speculative decoding over decoding
without it is S = τ / c, where τ is the tokens committed per request per decode step and c is
the time of one speculative step divided by the time of one step without speculation. Each
round finds which of three factors leaves the most room, works on that factor, and keeps a
change only on paired evidence. Keep notes in the work directory: every number, the command
that produced it, and every decision with its reason.

### 1. Split the requests

```bash
python skills/optimize-speculative-decoding/scripts/split_requests.py <requests.jsonl> \
  --out-dir <work>/requests --salt <study name>
```

When the requests are turns of multi-turn sessions, add `--group-field <session field>` so a
session's turns stay in one set. The sealed set (30 % of requests) stays unused until step 7.
Every other step uses the exploration set.

A run decodes for a few minutes. When the exploration set holds far more, fix one order of its
sessions whose prefixes keep the whole set's mix of context lengths, and give every arm at a
batch size the same prefix. Cap `max_new_tokens` at a high percentile of the set, so one very long
answer does not decide when a run ends.

**Check**: both files exist, their counts and sha256 are in the notes, with the order, the cap and
the prefix of each batch size.

### 2. Measure the starting point

A run is one engine and one client. Start `decoupled-engine` with the model, the draft
arguments of the starting configuration and `--sched-trace-dir <run>/trace`, without
`--disaggregation-mode`, so it serves requests by itself: these measurements are about one
engine at a fixed batch size, not the router. [`deploy-stack`](../deploy-stack/SKILL.md) covers
the image and the model paths; if the engine does not start, use
[`diagnose-run`](../diagnose-run/SKILL.md). Then, for each batch size:

```bash
python skills/optimize-speculative-decoding/scripts/fixed_batch_client.py --endpoint http://<host>:<port> \
  --requests <work>/requests/explore.jsonl --concurrency <B> --out <run> \
  --arm start --node <GPU node> --cycle 0 --trace-dir <run>/trace
python skills/optimize-speculative-decoding/scripts/decode_steps.py <run>/trace --concurrency <B> \
  --start <t_start> --end <t_end>      # from <run>/run.json
```

Repeat with speculation off: same model, GPUs and arguments, without the draft arguments. Its
step time is the denominator of c, and its outputs are the target model's own tokens for the
exploration set. When one engine serves several client runs, `POST /flush_cache` between them.

**Check**: for each batch size, `decode_steps.py` reports a few hundred plain steps or more,
and its τ agrees within a few percent with the accepted tokens per verify step that the client
prints from the engine's replies (the client counts every verify step, `decode_steps.py` only
full-batch steps).

### 3. Find the largest factor

The room left in a configuration is the product of three factors, measured at each batch size:

- **Drafting**: the largest S among draft settings divided by S now. Rerun the exploration set
  with the outputs of the run without speculation forced (`--forced-from <run without
  speculation>`), under each draft setting worth trying (more or fewer draft tokens, deeper or
  wider trees, other budgets), so every setting is measured on the same tokens. Compute each
  setting's S with its own step time: a setting that verifies fewer rows can have the larger S
  with a smaller τ. The engine rejects forced outputs for draft methods that do not support them;
  for those, measure each setting with paired live runs (step 5), whose outputs diverge.
- **Kernels**: the GPU busy time of one step divided by its floor. While the client holds the
  full batch:

  ```bash
  curl -X POST http://<host>:<port>/start_profile -H 'Content-Type: application/json' \
    -d '{"output_dir": "<run>/profile", "num_steps": 200, "activities": ["CPU", "GPU"]}'
  python skills/optimize-speculative-decoding/scripts/profile_steps.py <one GPU's trace file>
  python skills/optimize-speculative-decoding/scripts/byte_floor.py --model-dir <model> --requests <B> \
    --rows-per-request <target rows per request per step> --context-tokens <mean context> \
    --peak-tflops <datasheet> --peak-tbps <datasheet> [--kv-bytes-per-token ...] [--state-bytes-per-request ...]
  ```

  `/start_profile` returns at once and writes the trace after `num_steps` steps and an export:
  start it when the client still has at least that many full-batch steps to go, and keep the
  engine up until the file exists. Each script's `--help` says what its numbers mean and where
  its inputs come from. Give each kernel class that `byte_floor.py` does not model a floor by the
  same rule; for the all-reduces between GPUs the floor is the bytes each GPU moves over the link
  bandwidth ([references/cost-model.md](references/cost-model.md)). Check the class of the top
  kernels that `profile_steps.py` lists.
- **Overhead**: the wall time of the profiled steps divided by their GPU busy time, both from the
  same profile (`profile_steps.py` prints both): time in which the GPU waits for the CPU,
  launches or synchronization.

**Check**: a table per batch size with τ, S, the largest S of the draft settings, step time, GPU
busy time and floor per step, the three factors, and the largest one named.

### 4. Choose directions for the largest factor

| Largest factor | Where to look | Cheapest evidence |
|---|---|---|
| Drafting | draft length and tree shape; how a batch shares a budget of draft tokens; another draft method the engine has; proposals copied from the context | exact replay with forced outputs (step 3) |
| Kernels | the classes furthest above their floor | the kernel alone at the step's shapes on one GPU |
| Overhead | CPU work between steps, shapes without CUDA graphs, metadata rebuilt every step, synchronization | the idle gaps in the profile |

The regime decides which mechanisms pay: work added to every verify step helps a single request
and costs a full batch, while choosing which rows to verify helps a full batch
([references/cost-model.md](references/cost-model.md)). Work on several directions in parallel
where possible. Build a mechanism only when the evidence shows at least 3 % of room after its
own cost. Close a direction only after measuring the implemented mechanism, unless one
measurement costs more than 10 GPU-minutes. Every change goes behind a flag or environment
variable that is off by default.

**Check**: every direction has its evidence number and a decision (build or close) in the notes.

### 5. Promote with paired runs

Within one job on one GPU node, runs of the starting configuration (`--arm base`) and of the
change (`--arm <change>`) alternate, one cycle each (base, change, base, change, ...), each run
on a freshly started engine; a second node runs the same with the arms in the other order. A
pair is the base run and the change run of one cycle on one node.

```bash
python skills/optimize-speculative-decoding/scripts/paired_ab.py --base base --cand <change> <run dirs ...>
```

It prints the speed ratio of every pair, the mean ratio with its 95 % interval and per node, the
noise (base against base), the split of the ratio into τ and step time, and the decision:
promote when the lower end of the interval exceeds 1 % plus twice the noise. It refuses to
decide when a run has no partner or no plain steps.

- **Number of pairs.** Fix it before the first run of a comparison, from the noise of the
  starting point and the smallest ratio r worth promoting: n pairs with t × noise / √n below
  r − 1.01 − 2 × noise, where t is the 95 % t value for n − 1 degrees of freedom (3.2 for four
  pairs, 2.3 for nine), and at least two cycles on each of two nodes. Live runs diverge and
  have more noise. Adding pairs after a near miss decides on the data: judge the added pairs on
  their own.
- **One node at a time.** When the budget holds one node of the size a batch needs, run the
  second node's block after the first, with the arms in the other order. Compare single changes
  where a comparison is cheap, and only their stack where it is expensive.
- **Nodes that disagree.** When the per-node ratios differ by more than the noise while the base
  runs agree, report each node's ratio and look at the slower node before promoting.

**Check**: the base-against-base interval it prints contains 1, and the decision is in the notes.

### 6. Stack, tune kernels, repeat

Changes that each showed at least 3 % of room but did not pass promotion alone are stacked and
measured as one change with step 5. Then tune the kernels furthest above their floor; Nsight
Compute belongs here, once the factors say which kernels matter. After every promotion, return
to step 3: the largest factor changes.

**Check**: every round has its own factor table and the paired result of what it promoted.

### 7. Seal the result

Once, on the sealed set, at each batch size: step 5 with the starting configuration as base and
the final configuration as the change. Then check that the final configuration commits the
target model's tokens as faithfully as the starting one. Score the outputs of both sealed runs
on an engine without speculation, and compare. Under greedy decoding `--score-from` gives the
target's first choice at every position:

```bash
python skills/optimize-speculative-decoding/scripts/fixed_batch_client.py --endpoint <engine without speculation> \
  --requests <work>/requests/sealed.jsonl --concurrency <B> --score-from <sealed run> --out <sealed run>/scored
python skills/optimize-speculative-decoding/scripts/token_forcing.py --base <base scored dirs> --cand <final scored dirs>
```

Under the same-draws rule the reference is the token the target draws itself with the same key.
Replay each sealed run's outputs on the engine without speculation (`--forced-from <sealed run>
--keyed-seed`) and record its draw at every position, with the draw's margin over the runner-up.
If the engine cannot report its draws, that report is a change of its own, off by default, made
before this step. Write one line per request with `tok`, `top1` (the draw) and `gap` (the
margin); `token_forcing.py` reads that format.

Both rules must pass: the final configuration's mismatch rate is at most 0.5 percentage points
above the starting one's, and mismatches where the target is confident (a gap above 2: the
log-probability of its first choice over its second, or the draw's margin) are not more
frequent, both at 95 % confidence.

**Check**: `token_forcing.py` exits 0. If a rule fails, find the change that breaks it by
scoring the stack without each change, and remove that change.

### 8. Hand over

Each promoted change becomes its own pull request through [`check-pr`](../check-pr/SKILL.md),
with its paired result.

**Check**: every promoted change has a pull request, or the notes say why not.

## Returns

- The setup: model, draft method, GPU, GPUs per engine, batch sizes, decoding rule, the
  request set and the sha256 of both splits.
- Per round: the factor table of step 3, and a row per direction: the factor it targets, its
  evidence number, the decision, and the paired result (mean ratio, 95 % interval, noise, and
  the ratio per node when nodes disagree).
- On the sealed set, per batch size: the speedup over the starting configuration and over
  decoding without speculation, split into the τ ratio and the step-time ratio. These are
  ratios of decode-step throughput; the end-to-end effect on a served workload needs
  [`run-benchmark`](../run-benchmark/SKILL.md).
- Both lossless rules with their numbers.
- What was not run, and the decisions that are the owner's: which flags to turn on by
  default, and whether to run an end-to-end benchmark.
