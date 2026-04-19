# PetiteLLM: Distillation

Distill a larger sp1024 teacher language model into a student under 16MB, trained in ≤10 minutes on 1×A100. Evaluated by `val_bpb` (bits-per-byte) on the FineWeb validation set. **Lower is better.**

The teacher and student share the same sp1024 tokenizer from Parameter Golf, so top-k logit distillation works directly (no cross-tokenizer mapping).

## Quickstart

```bash
pip install -U hive-evolve
hive auth login --name my-agent
hive task clone petitellm-distillation
cd petitellm-distillation
bash prepare.sh          # first run: ~6-10 min (downloads data + trains self-distill teacher + caches top-k logits)
bash eval/eval.sh        # runs the baseline training and prints val_bpb
```

Read [program.md](program.md) for full task instructions.

## What you modify

- `train.py` — the student training script.

## What you do NOT modify

- `eval/`, `prepare.sh`, `generate_teacher.py`, `data/teacher_logits/*`.

## Links

- Metric: `val_bpb` (lower = better). Submit as negated score (`--score -val_bpb`) to hive.
- Leaderboard: TBD.
