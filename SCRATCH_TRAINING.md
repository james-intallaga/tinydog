# HybridLM scratch training

This is the random-initialization-only training path for the new
320,929,536-parameter `HybridLM`. It is separate from the live
301,325,312-parameter arithmetic Transformer. The hybrid architecture and
trainer are implemented, but the repository does not ship a trained hybrid
checkpoint or a training corpus, and the live service does not load it.

The trainer has no imported-weight option. It does not accept pretrained
Liquid, Qwen, DeepSeek, or other external model weights. Resume accepts only a
complete checkpoint created by this trainer, including its manifest, model,
optimizer, trainer, data cursor, tokenizer identity, and RNG state.

First follow `docs/scratch-data-pipeline.md` to create the frozen 32K scratch
tokenizer and audited train/validation data. Copy
`configs/hybrid-scratch.example.json`, set `tokenizer_path`, and copy the exact
`tokenizer_sha256` value from `tokenizer_meta.json`. A run and every resume
verify that digest before constructing the model. The all-zero digest in the
example file is intentionally invalid until replaced. Then run:

```sh
mlm-hybrid-train --config configs/my-hybrid-scratch.json
```

Resume from the last atomic checkpoint pointer:

```sh
mlm-hybrid-train \
  --config configs/my-hybrid-scratch.json \
  --resume checkpoints/hybrid-scratch/last.json
```

## Data

Each stage chooses `data_format: "audited_jsonl"` or
`data_format: "token_shards"`. Audited JSONL must be canonical output from the
data-preparation pipeline, including source, license, provenance, record ID,
and content checksum. A pretraining document looks like:

```json
{"type":"document","text":"A self-contained training document.","source":"collection","license":"CC-BY-4.0","provenance":{"file":"book.jsonl"},"record_id":"book-1","content_sha256":"..."}
```

Long documents are deterministically split into context-sized chunks. Every
chunk gets its own BOS/EOS boundary and padded batch row. Documents are never
concatenated into a shared attention stream.

Conversational SFT uses audited dialogue records:

```json
{"type":"dialogue","messages":[{"role":"user","content":"What is 12 * 8?"},{"role":"assistant","content":"96"}],"source":"consented-lessons","license":"CC-BY-4.0","provenance":{"session":"42"},"record_id":"lesson-42","content_sha256":"..."}
```

The frozen tokenizer inserts its native role ID and `END_TURN` ID around every
turn. Only assistant role, content, `END_TURN`, and a directly following final
EOS contribute to SFT loss. System, user, teacher, correction, memory, work,
and final-role turns remain context only.

With `data_format: "token_shards"`, point both stage paths at the prepared
directory (or its `shards` directory). Training reads the `train-*` shards and
evaluation reads the `validation-*` shards. Every shard index is bound to the
same tokenizer checksum before use.

Audited JSONL train and held-out paths must be distinct. Token shards use
separate train/validation indexes even when they share a directory. Their
SHA-256 hashes and the tokenizer artifact digest are recorded in
`provenance.json` and checked on resume.

## Baseline and safety

The baseline is AdamW with betas `(0.9, 0.95)`, weight decay `0.1`, global
gradient clipping at `1.0`, linear warmup, and cosine decay to a configurable
floor. BF16 parameters use FP32 Adam moments and FP32 master weights. Loss,
gradient, optimizer, and parameter finite checks occur before a step is
accepted. Non-finite or anomalous loss-spike steps are skipped, leaving the
last-good model and optimizer unchanged.

Checkpoints are written to immutable directories and published through atomic
`last.json` and `best.json` pointers. `metrics.jsonl` records training,
held-out evaluation, skipped updates, and checkpoints.

Muon, multi-token prediction, and reinforcement learning are deliberately not
part of this baseline. They should be evaluated later as isolated ablations
after the deterministic AdamW run is reproducible.

## What the stages do—and do not do

Stages execute in configuration order. A `pretrain` stage applies ordinary
next-token loss inside isolated BOS/EOS-bounded rows. An `sft` stage keeps
system/user/teacher/correction/memory/work/final-role text as context and
applies loss only to assistant output boundaries and content. Held-out loss is
evaluated periodically, and only this trainer's complete atomic checkpoints
can resume a run.

This trainer is text-only. It does not train the random waveform codec, video
encoder, or audio talker in `mini_arithmetic.multimodal`, and browser-captured
audio/video packets are not training examples. Multimodal training, alignment,
and inference integration remain future work; see
`docs/scratch-multimodal-scaffold.md`.
