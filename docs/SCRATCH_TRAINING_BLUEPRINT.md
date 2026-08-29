# MLM scratch-training blueprint

This blueprint applies to the new 320,929,536-parameter hybrid research model,
not the live 301,325,312-parameter arithmetic Transformer. The hybrid remains
untrained and is not loaded by the live service.

MLM is trained from random initialization. Liquid, Qwen, DeepSeek, and other
external checkpoints are neither loaded nor used as initialization. Published
work is used only to choose architecture, data, optimization, and evaluation
methods.

Tokenizer construction, corpus auditing/sharding, the text `HybridLM`, and the
deterministic pretrain/SFT trainer are implemented. No frozen tokenizer
artifact, corpus, completed foundation run, or trained hybrid checkpoint is
bundled. The audio/video model and browser transport are separate untrained
scaffolds, not additional completed training stages.

## Outcome order

The first target is a small model that can hold a basic conversation.  The
capability order is deliberately:

1. language completion and short-form coherence;
2. conversational turn taking and instruction following;
3. verified elementary arithmetic;
4. AMC-style problem solving;
5. olympiad work only after the earlier held-out gates pass.

Arithmetic-only training cannot create a conversational model from random
weights.  The base therefore needs a substantial clean language corpus before
conversational supervised fine-tuning (SFT) and arithmetic specialization.

## What is adapted from DeepSeek

These are process ideas, not copied weights or scale-specific hyperparameters.

- Require source, license, revision, and checksum provenance for every corpus.
- Remove templated or bulk-generated junk, exact duplicates, and near
  duplicates before assigning deterministic train, validation, and test
  splits.
- Keep fresh evaluation prompts outside every training and practice-data path.
- Batch and pad samples with unrelated documents and dialogues causally
  isolated. If true packing is added later, require an audited block mask.
- Use a stable AdamW baseline with beta1 0.9, beta2 0.95, weight decay 0.1,
  linear warmup, controlled decay, and global gradient clipping at 1.0.
- Record every run deterministically and save model, optimizer, scheduler,
  data cursor, random state, configuration, and data-manifest identity.
- Detect non-finite values and loss/gradient spikes before they corrupt the
  last known-good checkpoint.
- Use high-quality conversational SFT before any reinforcement-learning stage.
- Retain ordinary non-reasoning conversation during math specialization to
  avoid turning the model into a narrow answer generator.
- For arithmetic, prefer exact local rewards and rejection sampling over an
  unverifiable model judge.

DeepSeek-specific MoE routing, million-token sparse attention, mHC/Sinkhorn,
FP4/FP8 training, and datacenter parallelism are intentionally excluded.  Muon
and multi-token prediction are labelled experiments: neither is allowed into
the baseline until a controlled small-model A/B test improves held-out quality
per unit of wall-clock time without reducing stability.

## Stage 0: tokenizer and data contract

- Train and freeze one 32,768-ID byte-level BPE tokenizer.
- Preserve arbitrary UTF-8 through byte fallback; do not lowercase or destroy
  accents and mathematical symbols.
- Keep digits and common arithmetic operators atomic.
- Reserve explicit system, user, assistant, teacher, correction, memory, work,
  final-answer, and end-of-turn tokens.
- Store cleaned documents/dialogues in checksum-addressed, split-disjoint
  shards.  Never silently change a dataset underneath a resumed run.

## Stage 1: inexpensive architecture pilot (planned)

Train a scaled model before spending on the full hybrid. No dedicated pilot
configuration is currently shipped, so its exact shape and parameter count
must be defined, checked, and committed before this stage can run. Compare the
hybrid convolution/GQA backbone with an all-GQA model using the same tokenizer,
examples, token budget, optimizer, and seed family.

The pilot must demonstrate:

- finite, smoothly decreasing train and validation loss;
- reproducible resume behavior;
- better or equal held-out language and arithmetic loss per wall-clock hour;
- correct full-sequence versus cached decoding;
- no cross-sample leakage in packed data;
- acceptable Mac memory use and generation speed.

### Initial data candidates

Nothing is downloaded automatically.  Each candidate must be pinned to a
revision and pass the local provenance and filtering pipeline first.

- `HuggingFaceTB/smol-smoltalk` is an Apache-2.0 conversational subset made
  specifically for models below 1B parameters.  Its shorter examples are a
  better first SFT source for MLM than the full long-form mix.
- `OpenAssistant/oasst1` provides Apache-2.0 human conversation trees and is
  useful as a smaller independent conversational source.
- `HuggingFaceTB/cosmopedia` is Apache-2.0 synthetic educational text.  Use a
  quality-filtered sample, because repetitive templates are especially risky
  for a tiny model.
- `HuggingFaceFW/fineweb-edu` is an ODC-By educational web corpus subject to
  its dataset card and Common Crawl terms.  Stream only a curated pinned
  subset; do not mirror the multi-terabyte raw corpus onto this machine.

Dataset cards:

- https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
- https://huggingface.co/datasets/OpenAssistant/oasst1
- https://huggingface.co/datasets/HuggingFaceTB/cosmopedia
- https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

## Stage 2: 320,929,536-parameter base pretraining

Start from a freshly generated local random checkpoint.  A practical minimum
planning floor is about 6.4B tokens (roughly 20 tokens per parameter); 8–12B
clean tokens is a more credible target for this deliberately small deployment
model.  This is still far below the budgets used by modern production small
models, so dataset quality matters greatly.

Use a broad, licensed mixture rather than chat transcripts alone.  A starting
mixture for measurement—not a permanent truth—is:

- 60% curated general and educational prose;
- 20% natural dialogue and instruction-like text;
- 15% verified math and quantitative explanations;
- 5% small, high-quality code/structured text.

Sample by source rather than allowing the largest corpus to dominate.  Report
validation loss separately for general language, dialogue, and math.  Begin at
a modest sequence length, then extend only after the shorter-context run is
stable; long context is not a prerequisite for basic conversation.

## Stage 3: conversational SFT

Train on multi-turn examples with loss on assistant spans only.  Include short
greetings, questions, explanations, clarification, correction, refusal,
summarization, and stable persona/identity behavior.  Keep responses concise
enough for a roughly 321M model and include explicit end-of-turn targets.

Do not begin reasoning RL here.  First require frozen conversational tests to
show that MLM:

- answers instead of continuing the user's role;
- stops cleanly without repetition collapse;
- follows simple instructions and context from previous turns;
- admits uncertainty rather than fabricating on unknown factual questions;
- retains language coherence and basic safety behavior.

## Stage 4: arithmetic specialization

Continue training on verified arithmetic, explanations, corrections, and
retries while replaying conversational and general-language data.  Direct and
corrected sequences should both be learned:

```text
USER problem -> ASSISTANT correct answer
USER problem -> TEACHER correction -> ASSISTANT revised correct answer
```

Promotion requires exact generated-answer accuracy on a frozen unseen split,
not merely lower next-token loss.  It must also preserve conversational gates
and old-skill replay performance.

## Stage 5: later reasoning experiments

Only after stages 1–4 pass should we test verifier-guided reasoning or teacher
distillation.  Begin with arithmetic because answers are locally checkable.
Sample several candidate solutions, reject incorrect or malformed ones, and
train on verified successes.  Keep human/API provenance, provider terms, and a
held-out benchmark boundary.  Never train general model-judge feedback as
ground truth without independent validation.

## Primary references

- DeepSeek-V4 report: https://arxiv.org/html/2606.19348
- DeepSeek-V3 technical report: https://arxiv.org/html/2412.19437
- DeepSeek-R1 report: https://arxiv.org/html/2501.12948
- DeepSeek LLM repository and data notes:
  https://github.com/deepseek-ai/DeepSeek-LLM
- LFM2.5 model architecture: https://huggingface.co/LiquidAI/LFM2.5-350M
- Official MLX LFM2 implementation:
  https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/lfm2.py
- Chinchilla compute/data scaling study: https://arxiv.org/abs/2203.15556
