# Query-access interference probe (R0–R3)

This standalone experiment measures sequential LoRA query-access interference. It uses
EasyEdit's existing `BaseEditor.edit` and `execute_lora`; neither implementation is
modified. It implements measurement and linear prediction only. No anti-forgetting
method or layer intervention is included.

- R0: freeze facts/order, initialize a zero-output LoRA adapter, record all base scores.
- R1: write exactly one fact per `editor.edit(..., sequential_edit=True, verbose=False)`;
  save its CPU FP32 delta and the full previously written fact trajectory.
- R2: measure current-state and frozen-at-write gap gradients, their dot products with
  each later write, actual gap change, and the exact layer decomposition.
- R3: compare cumulative predictors on final stream/fact rows, with leave-stream-out
  linear calibration and stream bootstrap confidence intervals.

## Data contract

Provide one JSON object per line with all seven fields:

```json
{"fact_id":"example-1","write_prompt":"The capital of France is","target_new":"Lyon","access_questions":["Which city is the capital of France?","Name France's capital city."],"storage_prompt":"The capital of France is","support_context":"For this task, the capital of France is Lyon.","competitor_answers":["Paris","Marseille"]}
```

This is a counterfactual **format example**, not a supplied experimental dataset.
Prepare at least 20 reviewed facts. Keep `storage_prompt` close to the training
recitation, access questions held out, support context explicitly informative, and
competitors plausible (include the original preferred answer). The loader rejects
missing/empty fields, duplicate fact IDs, duplicate questions/competitors, a target
also listed as competitor, unresolved `{}` write templates, and verbatim reuse of
a write/storage prompt as an access question. Semantic held-outness, informative
context and competitor plausibility require dataset review.

The first N input records form a fixed cohort; the seed shuffles only that cohort.
Each seed starts from a newly loaded base model and a newly initialized adapter.
Every stream saves its full records, seed, order, data hash, effective hyperparameters,
score conventions, package versions and Git commit in `seed_SEED/stream_manifest.json`.
Do not tune data, thresholds, competitors or hyperparameters on held-out streams.

## Server execution

Run from the repository root in the existing EasyEdit CUDA environment. Use the
repository's installed dependencies; the module adds no production dependency.
A fast tokenizer supporting character offsets is required. The default configuration
is `hparams/LoRA/qwen2.5-7b.yaml`. The runner explicitly overrides only
`lora_dropout=0.0`, `batch_size=1`, and the requested device/model path. Other LoRA
settings, including 60 steps, rank 8 and learning rate 0.005, come from that YAML.
The effective configuration is saved. No optimizer step is used for initialization.

First run **5 facts, 1 seed**:

```bash
python -m experiments.query_access.run_probe \
  --data /path/to/reviewed_facts.jsonl \
  --hparams hparams/LoRA/qwen2.5-7b.yaml \
  --model-path /path/to/Qwen2.5-7B-Instruct \
  --output /path/to/results/query_access_smoke \
  --stage smoke --seeds 41 --device 0 \
  --tau-storage 0.0 --tau-context 0.0 --max-new-tokens 32
```

Omit `--model-path` to use the YAML model identifier. For local paths, preserve
`qwen2` in the directory name: EasyEdit dispatches its model loader by name.
Choose generation length sufficient for complete answers; truncated generations
count as EM failures. The margins do not use generation.

Check smoke output: **20 trajectory rows** (5 base + 15 post-write), **10 interference
rows**, **5 delta files**, **5 frozen gradient files**, and per-layer records summing
to each full `I_state`. Files must contain finite scores and gradients. The manifest
and `run_completed.json` identify a complete run. One stream has no out-of-stream
prediction estimate or bootstrap interval; these fields intentionally remain null
or `insufficient_streams`.

Then run **20 facts, at least 3 stream seeds**:

```bash
python -m experiments.query_access.run_probe \
  --data /path/to/reviewed_facts.jsonl \
  --hparams hparams/LoRA/qwen2.5-7b.yaml \
  --model-path /path/to/Qwen2.5-7B-Instruct \
  --output /path/to/results/query_access_gate1 \
  --stage gate-1 --seeds 41 42 43 --device 0 \
  --smoke-output /path/to/results/query_access_smoke \
  --tau-storage 0.0 --tau-context 0.0 --max-new-tokens 32
```

Gate-1 requires a completed smoke run. For three 20-fact streams expect **690
trajectory rows**, **570 interference rows**, and **60 delta files**. The CLI has
no 100-fact mode. Output directories must be new or empty; partial runs are preserved
for inspection and must be restarted in a new directory. Run completion is marked
only after all streams and summary outputs finish. A fresh model is loaded for each
seed, so seeds are sequential on one device, not distributed training.

## Score and gradient definitions

`mean_answer_logprob` tokenizes `prompt + " " + answer`, uses shifted causal logits,
and averages log probabilities over answer tokens only. Prompt and special tokens
are masked using offsets, with no truncation. Leading separator whitespace attached
to an answer token is allowed; a token spanning both prompt characters and answer
characters raises an error rather than conditioning on part of the answer.
The concatenation matches EasyEdit LoRA; offset masking is explicit instead of
relying on separately tokenized prompt lengths at BPE boundaries.

The smooth margin is target mean log probability minus `logsumexp` of competitor
mean log probabilities. Competitor sets are fixed for a fact across every evaluation.

- `access_margin`: mean margin over held-out access questions (A).
- `storage_proxy_margin`: canonical/recitation prompt margin (S), a storage **proxy**.
- `context_supported_margin`: mean margin of `support_context + "\n" + question` (C).
- `access_gap`: S − A (D).
- `access_specific_failure`: A ≤ 0 and S ≥ `tau_storage` and C ≥ `tau_context`.

Threshold defaults of zero are prespecified decision margins, not fitted estimates.
Change them before running an experiment; any data-driven choice must use training
streams only. Keep the same choice across streams in one run.

Auxiliary `access_em`, `storage_em`, `context_em` use deterministic greedy generation,
full-string equality after casefold/strip/whitespace normalization, and average over
questions where relevant. No punctuation or article stripping is performed.
`latest_write_intrusion` is the fraction of access questions answered exactly with
the latest write's target. It is null for base rows, the newest fact itself, or facts
whose target equals the newest target; these cases cannot identify intrusion.
`write_em` is recorded on each fact's own write prompt immediately after its write.
Base rows have `write_idx=-1`, `age=null`; post-write age is `write_idx-eval_fact_idx`.

Measurement clears residual training `.grad`, disables training checkpointing, and
uses eval mode. `torch.autograd.grad` computes storage and each access contribution
separately and accumulates CPU FP32 named gradients; it never calls `.backward()`.
Every completed measurement asserts no parameter has a `.grad` buffer. EasyEdit
reenables its normal checkpointing at the next write. Only trainable `lora_`
parameters are included; a trainable non-LoRA parameter is an error.

A fact's frozen gradient is measured just after its own write. Before every later
write, the current-state gradient is measured again. After writing:

```text
I_state = <current gap gradient, write delta>
I_fixed = <frozen gap gradient, write delta>
actual_delta_D = gap_after - gap_before
```

Snapshots, deltas and saved gradients are CPU FP32. Dot products and norms reduce
in FP64. Names and shapes must match. Layer names use `layers.<N>`; unmatched names
are kept as `other`. The sum of layer interference must differ from the full dot
product by less than 1e-6. No flattened GPU gradient history is retained.

## Outputs and statistics

```text
output/
  query_access_trajectory.jsonl
  query_access_interference.jsonl
  query_access_layer_interference.jsonl
  query_access_summary.json
  query_access_summary.csv
  run_completed.json
  seed_SEED/
    stream_manifest.json
    completed.json
    deltas/write_XXXX.pt
    fixed_gradients/fact_XXXX.pt
```

Each delta file stores `delta` plus `fact_id`, `write_idx`, seed, total/per-layer
norms, `num_steps`, `lr`, and `rank`. Current-state gradients temporarily use
`state_gradients/`; files are removed after contraction, while interrupted runs
retain them. Disk space for fixed gradients and deltas scales with stream length
and adapter size. This probe intentionally recomputes every historical gradient,
so server runtime scales quadratically with fact count.

Cumulative state interference, frozen interference and delta norm sum **later writes
only** (s > i), starting at zero after the fact's own write. Summary prediction uses
one final observation per stream/fact with at least one later write (`age > 0`).
The newest fact is excluded from forgetting prediction and final failure/retention
rates because it has no later-write exposure.

Summary definitions:

- `immediate_edit_success`: mean immediate **write-prompt EM**.
- `final_retention`: mean final access EM among exposed facts, without conditioning
  on initial success; `immediate_access_em` is also reported for comparison.
- `access_specific_failure_rate` and `latest_write_intrusion_rate`: final exposed facts.
- `rho_state_one_step` / `rho_fixed_one_step`: Spearman across stream/fact **means**
  of paired later-write predictions and actual gap changes. This deliberately uses
  one observation per stream/fact; individual transitions remain in the raw JSONL.
  `one_step_transition_diagnostics` additionally reports paired-transition Pearson
  and Spearman with whole-stream bootstrap, retaining repeated-step dependence.
- `rho_cumulative_*_final_access`: descriptive Spearman against final access margin.
- `delta_rho_state_minus_fixed`: signed difference of these final-access correlations.
  Since positive gap interference may predict lower access, a more negative rho can
  indicate stronger association; a positive difference does not inherently mean better.

The summary includes Pearson, Spearman and bootstrap 95% intervals for all three
cumulative predictors against final access margin, gap and strict failure. Bootstrap
resamples **whole streams** (all facts stay together), using 1,000 draws by default.
At least three streams and sufficient valid nonconstant resamples are required;
undefined correlations/intervals are JSON null, never NaN. With only three streams,
intervals are exploratory and have limited cluster diversity. These intervals are
conditional on the fixed cohort and cover stream-order variation, not new fact cohorts.

`leave_stream_out` fits a separate intercept + linear slope for each predictor and
target using only final rows from other streams. Feature scaling is fit on those
training streams. For strict failure this is a linear probability score; its binary
cutoff maximizes training accuracy only. Saved predictions include training seeds,
coefficients, scaling and threshold, along with held-out Pearson, Spearman, MSE and
classification accuracy. The same stream never supplies training and test rows in
a fold. These held-out results are the prediction check; pooled correlations alone
are descriptive. Bootstrap association intervals do not quantify fitted-model
uncertainty. No nonlinear predictor, KL or optional cosine baseline is implemented.

## Offline tests

```bash
python -m pytest tests/test_query_access_probe.py -q
```

Tests use a randomly initialized tiny Llama created from configuration, a tokenizer
created locally, PEFT, and a mock editor. They download no model or tokenizer. They
cover zero effective initialization, first-write delta, reconstruction, score purity
and answer masking, gap identity, named dot products, exact layer sum, changing
state gradients, deterministic order, empty `.grad`, five-fact file/cumulative
integrity, leave-stream-out leakage, and whole-stream bootstrap. A compatibility
test loads the unmodified production `execute_lora` function from its source (without
importing unrelated EasyEdit components) and checks adapter reuse and gradient
measurement after its actual optimizer/checkpointing path.

Local tests establish implementation behavior. Real Qwen smoke and Gate-1 must run
on the server before reporting empirical interference or forgetting results.
