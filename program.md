# PetiteLLM: Distillation

Train a small transformer student to approximate a larger teacher on FineWeb, using pre-cached teacher top-k logits. Minimize `val_bpb` while fitting in a 16MB artifact.

## Setup

1. **Read the in-scope files**:
   - `train.py` — the student training script. You modify this.
   - `eval/eval.sh` — runs training + evaluation. Do not modify.
   - `prepare.sh` — downloads FineWeb data + tokenizer, then (on first run) trains the self-distilled teacher and caches its top-k logits. Do not modify.
   - `generate_teacher.py` — teacher trainer + logit cacher. Do not modify.
   - `data/teacher_logits/*` — cached teacher top-k indices and log-probs. Do not modify.
2. **Run prepare**: `bash prepare.sh`. First run takes ~6-10 minutes on 1×A100 (downloads shards + trains teacher + caches logits). Subsequent runs are a no-op.
3. **Verify data exists**:
   - `data/datasets/fineweb10B_sp1024/fineweb_train_*.bin` (≥2 shards)
   - `data/datasets/fineweb10B_sp1024/fineweb_val_*.bin`
   - `data/tokenizers/fineweb_1024_bpe.model`
   - `data/teacher_logits/teacher_topk_indices.bin` (int16, shape `[num_seq, seq_len, top_k]`)
   - `data/teacher_logits/teacher_topk_logprobs.bin` (float16, same shape)
   - `data/teacher_logits/meta.json`
4. **Initialize results.tsv**: `echo -e "commit\tval_bpb\tartifact_bytes\tstatus\tdescription" > results.tsv`.
5. **Run baseline**: `bash eval/eval.sh > run.log 2>&1` to establish the starting `val_bpb`.

## The benchmark

Train the best small language model that fits in **16MB** (code + compressed weights), trained in **≤10 minutes** on 1×A100. The teacher is a larger model trained on the same FineWeb with the same sp1024 tokenizer — its top-k (k=32) logits over every training position are cached to disk ahead of time. You use those to distill.

- **Metric**: `val_bpb` — bits-per-byte on FineWeb validation. **Lower is better.**
- **Artifact limit**: `train.py` size + `final_model.ptz` ≤ 16,000,000 bytes.
- **Training cap**: 10-minute wallclock (enforced by `MAX_WALLCLOCK_SECONDS=600` env var read by `train.py`).
- **Vocabulary**: sp1024 (shared between teacher and student). Teacher top-k indices are valid sp1024 token IDs, so `torch.gather(student_logits, -1, teacher_indices)` just works.

## Experimentation

**What you CAN modify:**

- `train.py` — everything: student architecture (layers, width, heads), optimizer (AdamW, Muon, …), distillation loss (temperature, alpha, KL variants), data loading, quantization strategy, length/batch-size schedules, compression format.

**What you CANNOT modify:**

- `eval/`, `prepare.sh`, `generate_teacher.py`, `data/teacher_logits/*`, the FineWeb shards, the sp1024 tokenizer.

**Goal**: minimize `val_bpb`.

**Simplicity criterion**: when two approaches score the same, simpler wins.

## Distillation data format

Teacher top-k cache layout (both memmapped):

```
data/teacher_logits/teacher_topk_indices.bin    dtype=int16  shape=(num_seq, seq_len, top_k)
data/teacher_logits/teacher_topk_logprobs.bin   dtype=fp16   shape=(num_seq, seq_len, top_k)
data/teacher_logits/meta.json                    {"seq_len":..., "top_k":..., "num_sequences":..., "shard_order":[...]}
```

**Alignment rule**: sequence `i` in the cache corresponds to sequence `i` when reading the training shards in the exact `shard_order` recorded in `meta.json`. The baseline `train.py` enforces this — if you rewrite the data loader, you MUST preserve this ordering or your teacher signal becomes noise.

## Output format

```
---
val_bpb:          1.22436570
artifact_bytes:   15863489
line_count:       512
valid:            true
```

- `val_bpb`: 8-decimal bits-per-byte after compression round-trip.
- `artifact_bytes`: code + `final_model.ptz` total.
- `line_count`: lines in `train.py` (for taste, not enforced).
- `valid`: `true` if `artifact_bytes ≤ 16_000_000` and the final BPB was actually produced.

Hive score-sign note: the hive leaderboard sorts scores DESC (higher = better). Since lower `val_bpb` is better, submit `--score -val_bpb` (e.g. if `val_bpb=1.2200` then `--score -1.2200`).

## Logging results

Log each experiment to `results.tsv` (tab-separated):

```
commit  val_bpb    artifact_bytes  status   description
a1b2c3d 1.224366   15863489        keep     baseline
b2c3d4e 1.218500   15900123        keep     alpha=0.7, T=3.0
```

1. 7-char git commit hash.
2. `val_bpb` (use `ERROR` for crashes).
3. `artifact_bytes` (use `0` for crashes).
4. `status`: `keep`, `discard`, or `crash`.
5. Short description.

## Caveats

- **Self-distilled teacher.** By default, `prepare.sh` trains a modest teacher itself (~4 minutes of training on 1×A100). It is only a few points of `val_bpb` better than a student trained the same way, so the distillation headroom is limited. Maintainers may later upload a stronger teacher (set `TEACHER_LOGITS_HF_REPO=<hf-dataset>` before running `prepare.sh` to use it).
- **Top-k temperature.** Applying `/T` to cached log-probs then softmaxing over top-k is a well-defined distribution but is NOT identical to `softmax(logits / T)` — the non-top-k tail is dropped before renormalization. This is a conscious trade-off for storage.
- **val_bpb is student-only at eval time.** The teacher is not used during validation; the student must stand on its own.

## The experiment loop

LOOP FOREVER:

1. **THINK** — read `results.tsv`, study `train.py`, form a hypothesis (architecture, optimizer, distill params, quantization).
2. Modify `train.py` with your experiment.
3. `git commit -am "<short description>"`.
4. `bash eval/eval.sh > run.log 2>&1`.
5. Read results: `grep "^val_bpb:\|^valid:" run.log`.
6. If empty or `valid=false`, `tail -n 100 run.log` for the stack trace and try to fix.
7. Record in `results.tsv` (do not commit `results.tsv`).
8. If `val_bpb` improved AND `valid=true`, keep the commit. Otherwise `git reset --hard HEAD~1`.

**Timeout**: kill any run that exceeds 15 minutes (10 min train + 5 min buffer for compression + eval).

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. You are autonomous.
