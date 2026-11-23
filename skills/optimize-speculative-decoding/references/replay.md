# Replay

The rules every run of the study follows: development runs, comparisons, lossless-check runs and
sealed finals. A run that breaks one of them measures nothing.

## Rules

1. **The engine.** One `decoupled-engine` with `--sched-trace-dir <run>/trace`, serving requests by
   itself, without `--disaggregation-mode`, on the same clock as its client.
2. **The client.** Every run is sent by `fixed_batch_client.py`:

   ```bash
   python skills/optimize-speculative-decoding/scripts/fixed_batch_client.py --endpoint http://<host>:<port> \
     --requests <requests file> --concurrency <B> --out <run> --arm <arm> --node <slot> --cycle <k> \
     --trace-dir <run>/trace --max-new-tokens-cap <cap> [--sampling '<JSON>'] \
     [--session-field <task field> [--session-copies <C>]] [--logprobs <K>]
   ```

   It writes every request's output and latency to `<run>/outputs.jsonl`, and the arm, slot, cycle,
   batch size, window, request file's sha256, sampling and cap to `<run>/run.json`. Decoding is greedy
   unless `--sampling` says otherwise; `--max-new-tokens-cap` caps every output's length. A request the
   engine aborted counts as failed, and a run with a failed request exits 1. A paired run is two runs
   at the same time on two slots.
3. **The requests.** One JSON object per line: `id`, the prompt as `input_ids` (token ids) or `text`
   (with `image_data` for images), `max_new_tokens`, and the group field of the gain. Tasks of several
   turns also carry the task field and each turn's recorded answer: `output_text` for a text prompt,
   `output_ids` for a token prompt.
4. **The batch size.** B is the number in flight per engine in a closed loop. For independent
   requests, the client keeps B requests in flight and starts the next one when one finishes. For
   tasks of several turns (a robot's episode, an agent's session), where each turn is sent only after
   the previous one is answered, B counts tasks: the client keeps B tasks in flight and starts the next
   task when one ends. The client refuses a request set with fewer requests, or tasks times copies,
   than B: it could not keep B in flight. The engine's batch, the requests it decodes together in one
   step, varies between 1 and B: a task between two turns has no request in the engine.
5. **Tasks of several turns.** Pass `--session-field <task field>`: the requests that share the
   field's value are the turns of one task, in file order. The client sends a task's turns one after
   another, each after the previous one returned, and a failed turn ends its task. A turn's prompt is
   the prompt the previous turn sent, then the answer the engine gave it (special tokens kept, the stop
   token dropped), then the turn's new part: its recorded prompt minus the previous turn's recorded
   prompt and answer. The client refuses a file in which a turn's recorded prompt does not extend the
   previous turn's prompt and answer. Every task has its own cache key, so tasks share no cached KV or
   recurrent state. When B exceeds the number of tasks, pass `--session-copies C` to run the task list
   C times; every copy has its own cache key, and copy k ≥ 1 of a request gets the id `<id>#c<k>`.
6. **The split.** `scripts/split_requests.py` assigns each request to the development file or a sealed
   part by a fixed hash of its key and the study's salt. For tasks of several turns pass
   `--group-field <task field>`, so a task's turns stay in one file. `--sealed-parts R` makes one sealed
   part per planned round. The script refuses a split in which one prompt lands in two files.
7. **Scoring.** The lossless checks score a run's outputs with `--score-from <run>` (lossless.md); the
   client rebuilds each turn's prompt from the run's own answers as the run built it, with the task
   field and copies read from the run's `run.json`, and refuses a request file other than the run's.
