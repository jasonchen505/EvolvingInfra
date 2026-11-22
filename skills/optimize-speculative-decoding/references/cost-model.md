# Cost model of speculative decoding

The method of SpecPilot ([arXiv 2611.02501](https://arxiv.org/abs/2611.02501)), as the
[skill](../SKILL.md) uses it. Every quantity is measured on one engine, at one batch size B,
on the same requests.

## Speedup

- **τ(N)**: tokens committed per request per decode step with draft setting N: the accepted
  draft tokens plus the one token the target model adds after them. Without speculation τ = 1.
- **c(N) = T_step(N) / T_AR**: the time of one plain decode step with setting N over the time of
  one plain decode step without speculation. A plain step is a decode step at the full batch
  that follows another decode step (`scripts/decode_steps.py`).
- **S(N) = τ(N) / c(N)**: decode throughput relative to decoding without speculation.
- Between two configurations at the same batch size, the speedup is
  (τ_new / τ_old) × (T_step,old / T_step,new). `scripts/paired_ab.py` reports both parts.

## The three factors

The room left in a configuration is

    (S_best / S) × (T_prof / T_floor) × (T_wall / T_prof)

- **S_best / S**, drafting. S_best is the largest S among the draft settings, each measured on
  the same tokens as S with its own step time; a setting that verifies fewer rows can have the
  larger S with a smaller τ. SpecPilot records the draft model's top candidates at every
  position of the target model's own outputs, rebuilds the best tree of each size offline, and
  accepts the longest path that matches the target's tokens, plus one token. The engine does not record
  candidates; forced outputs (`--forced-from` of `scripts/fixed_batch_client.py`) give the same
  comparison for the trees the engine can build. Recording candidates is a change of its own,
  worth making when trees the engine cannot build may matter.
- **T_prof / T_floor**, kernels. T_prof is the GPU busy time of one step in a profile, drafting
  included (`scripts/profile_steps.py`). T_floor is the sum over kernel classes of
  max(FLOPs / (0.6 × peak FLOPs), bytes / (0.9 × peak bandwidth)), where the bytes are what the
  step must move: every weight once, only the experts the step's rows reach in a mixture of
  experts, the KV cache of every request, the recurrent state read and written
  (`scripts/byte_floor.py`). With tensor parallelism over G GPUs, each GPU reads 1/G of the
  weights and 1/min(G, KV heads) of the KV cache: past G = KV heads the heads are copied.
  Communication has a floor by the same rule: an all-reduce of M bytes per GPU moves about
  2 (G − 1) / G × M per GPU in two shots (reduce-scatter, then all-gather) and (G − 1) × M in one
  shot, over the link bandwidth. Which algorithm runs depends on the message size, and the size
  at which the engine switches is a setting worth measuring on each interconnect.
- **T_wall / T_prof**, overhead: the wall time of the profiled steps over their GPU busy time,
  both from one profile: time in a step that no GPU work covers.

A round works on the largest factor. Recompute all three after every promotion.

## Rules

- **Evidence before building.** Check a direction's room with the cheapest evidence first:
  arithmetic on bytes and FLOPs, exact replay, the kernel alone, one GPU. Build only when the
  room is at least 3 % after the mechanism's own cost. Close a direction only after measuring
  the implemented mechanism, unless one measurement costs more than 10 GPU-minutes.
- **Promotion.** Paired runs: the same requests, the two arms alternating on the same GPUs, on
  two nodes. Promote when the lower end of the 95 % interval of the mean speed ratio exceeds
  1 + 0.01 + 2 × noise, where the noise is the root mean square of (ratio - 1) between runs of
  the base arm.
- **Stacking.** Changes that each show at least 3 % of room but miss promotion alone are
  measured together as one change.
- **Lossless only.** Under greedy decoding every committed token is the target model's most
  likely token. Under sampling, the owner's rule: the token the target draws itself with the
  same key, or a draw from the target's distribution. Every change is behind a flag that is off
  by default.
- **Sealed check.** On requests not used before, with the outputs of both systems scored against
  the target model's reference (its first choice, or its own keyed draw): the mismatch rate, compared within bins of the target's confidence and weighted
  by the base's bins, rises by at most 0.5 percentage points; mismatches where the target's
  first choice leads its second by more than 2 in log-probability (or the keyed draw leads by
  that margin) are not more frequent. Both at 95 % confidence (`scripts/token_forcing.py`).

## Which mechanisms pay depends on the batch

- Work added to every verify step, such as more draft tokens per request, raises τ and suits a
  single request; at a full batch it multiplies the rows the target model processes.
- Choosing which rows to verify, such as one budget of draft tokens shared by the batch or a
  draft length per request, suits a full batch.
- In a mixture of experts, extra rows reach more experts, and each new expert costs its weight
  bytes, until every expert is reached.
