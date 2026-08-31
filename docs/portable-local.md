# Run and train tinydog.ai on your own computer

tinydog.ai is intended to remain private and local. The setup command installs
program dependencies only. It does **not** download pretrained tinydog
language-model weights, upload conversations, or connect the learner to a
central training service. The web interface bundles Silero VAD solely for
speech/silence detection; that small pretrained detector does not transcribe,
answer, or train tinydog.

## The short path

Install Python 3.10 or newer and Node.js 22.13 or newer (a currently supported
LTS release is recommended), then download or clone
this repository.

Choose a model size from a numbered menu—no source code or configuration-file
editing is required:

```bash
./tinydog.sh model
```

On Windows use `tinydog.cmd model` or `.\tinydog.ps1 model`. Choose 1–6 for
20M, 50M, 80M, 100M, 200M, or 300M. The launcher remembers the choice. A
direct form such as `tinydog.cmd model 50m` works too.

On macOS or Linux, open a terminal in this folder and run:

```bash
./tinydog.sh start
```

On Windows, double-click `tinydog.cmd` or run:

```bat
tinydog.cmd start
```

PowerShell users can instead run:

```powershell
.\tinydog.ps1 start
```

The first start creates a private `.venv`, installs the matching compute
engine and the local web interface, and then opens the services. Later starts
reuse that setup. Open <http://localhost:3000> and press `Ctrl+C` in the
terminal when finished.

The production screen uses bundled local fonts and disables framework
telemetry; starting an already-installed learner does not contact a font host.

To install without starting, replace `start` with `setup`. To see what the
computer can use, run `doctor`.

## Automatic device choice

| Computer | Automatic engine | Accelerator reported when available |
|---|---|---|
| Apple Silicon Mac | MLX | Metal |
| Windows or Linux with NVIDIA GPU | PyTorch | CUDA |
| Linux with a compatible AMD PyTorch build | PyTorch | ROCm |
| Windows or Linux with a compatible Intel build | PyTorch | XPU |
| Mac without MLX but with PyTorch | PyTorch | MPS |
| Other supported computer | PyTorch | CPU |

On a fresh installation, `auto` chooses the 20,003,456-parameter profile on
every device. Accelerator detection alone cannot safely estimate free memory.
An existing installation that already has the earlier 29.85M `portable-tiny`
bundle stays on it, so an update never appears to erase that owner’s learning.

Choose and remember a size with:

```bash
./tinydog.sh model 20m
./tinydog.sh model 50m
./tinydog.sh model 80m
./tinydog.sh model 100m
./tinydog.sh model 200m
./tinydog.sh model 300m
```

Then start normally. `start --model-size 50m` is also available as a one-time
override and does not change the remembered choice.

Approximate complete-update planning ranges are 0.6–1.2 GB (20M), 1.2–2.5 GB
(50M), 1.8–3.5 GB (80M), 2.3–4.5 GB (100M), 4.5–8 GB (200M), and 7–12 GB
(300M). They are estimates, not guarantees. PyTorch expands BF16 bundles to
FP32 compute, and a personal update also needs a separate training model,
gradients, AdamW state, activations, and temporary publication space while the
live model keeps answering. Use 20M if there is any doubt.

The older `tiny` (29.85M) and `standard` (320.93M) choices remain valid only
for compatibility. They are different architectures, not aliases for 20M and
300M. Switching size selects a separate brain and separate learned lineage;
it does not convert or delete another size, and switching back resumes it.

This release does not yet train inside Android, iOS/iPadOS, or a browser. A
Chromebook may use the 20M CPU path through its Linux development environment
when that environment provides the required Python and Node.js versions.

The default PyTorch package is installed from its normal package source.
People who need a platform-specific CUDA, ROCm, or XPU build can supply the
official PyTorch package index for their machine:

```bash
./tinydog.sh setup --backend torch --torch-index-url <official-wheel-index>
```

The device report never calls a GPU available merely because PyTorch is
installed. It separately probes CUDA/ROCm, XPU, MPS, and finally CPU.

## Default learning mode and status

The portable service defaults to `MLM_PERSONAL_LEARNING_MODE=foundation`.
`./tinydog.sh start` and the Windows launchers pass `.env` into the service;
the equivalent direct command is `tinydog-serve --learning-mode foundation`.
Use `MLM_PERSONAL_FOUNDATION_AUTO=0` to keep inference running without starting
new automatic updates. `MLM_PERSONAL_FOUNDATION_DIR` changes the private
checkpoint root, while `--personal-root` overrides it for a direct server run.

`GET /status` exposes the implemented boundary rather than a generic “trained”
label. Useful fields include `portable.learning_mode`,
`learning.parameter_scope`, `learning.qualifying_pending`,
`learning.lesson_threshold`, `learning.monitoring_is_holdout`,
`learning.monitoring_is_promotion_gate`, `learning.promotion_policy`,
`learning.active_foundation_id`, and `learning.rollback_available`. The model
section reports `all_foundation_parameters_optimizer_eligible` and
`optimizer_state_continuity`. The live inference model is frozen between swaps;
`learning.foundation_trainable` describes the isolated training copy.

For compatibility testing only, set `MLM_PERSONAL_LEARNING_MODE=adapter` or run
`tinydog-serve --learning-mode adapter`. This selects the old v1 adapter path,
not a migration or merge into the default foundation lineage.

## What is portable and what is not yet

- The bootstrap, web interface, conversation database, tokenizer, portable
  text model, cached inference, and five-turn private full-foundation training
  run through MLX or PyTorch now.
- Every five distinct eligible trusted typed turns starts an asynchronous run.
  All five human targets and a deterministic sample of up to eight older local
  targets are optimization data. Every canonical foundation parameter is
  optimizer-eligible, although sparse or zero gradients mean that one short run
  does not guarantee that every scalar changes.
- One deterministic member of those same five targets is measured before and
  after as clearly labelled in-sample telemetry. It is not a holdout and never
  gates promotion. Quality can improve or regress; only hard technical checks
  such as identity/schema/checksum compatibility, finite training, changed
  weights, and a successful forward smoke test block activation.
- Optimizer state resets for every five-turn run in v1. The service reports
  `optimizer_state_continuity: false`; it does not claim that AdamW moments are
  carried from one conversational cycle to the next.
- Human-authored targets must fit intact inside the bounded personal
  training window. Oversized targets are rejected before the message ledger is
  changed, so one long turn cannot block every later five-turn cycle.
- Training uses an isolated model while the old checkpoint continues answering.
  A technically valid complete bundle is checksum-verified and activated in
  one swap. Immutable base, current, and previous bundles form the private
  lineage; `/rollback` returns to the previous bundle or the base.
- Legacy v1 adapters are an explicit, separate compatibility mode. Their
  artifacts cannot be merged into or rebound as v2 full-foundation checkpoints.
- The browser microphone/camera and WebRTC transport are cross-platform.
- The scratch-native neural audio/video model is still implemented with MLX.
  On Windows and Linux, transport compatibility alone does not turn it into a
  trained native voice model.
- Raw microphone audio still has no trustworthy semantic transcript. It must
  not be silently labelled as conversational language training. Until a local
  speech-understanding path is trained or added, typed dialogue is the honest
  zero-configuration conversational training path; semantic voice learning
  remains an MLX research gap.
- Training from random initialization remains real training: a fresh model
  will initially produce nonsense and will not become coherent from five turns.
  It needs substantial local data, evaluation, and time.

The original arithmetic MLX service remains available explicitly for
development:

```bash
./tinydog.sh start --legacy-mlx
```

There is no silent fallback to that service on Windows or Linux.

## Maintainer installation choices

The base Python package no longer depends on MLX. Compute engines are separate
extras:

```bash
python -m pip install -e '.[mlx,realtime]'  # Apple Silicon
python -m pip install -e '.[torch]'         # Windows/Linux/portable Torch
```

The MLX extra also has a macOS/arm64 package marker, so ordinary package
resolution on Windows or Linux does not attempt to install an unsupported MLX
wheel. `safetensors` is an explicit backend-neutral dependency so checkpoint
files have the same interchange format on both engines.

## Explicit full-foundation training with PyTorch

Five-turn talk-to-train now updates the complete foundation, but it is a
deliberately narrow one-cycle recipe over five new targets plus bounded replay.
It resets optimizer state each cycle and has no quality promotion gate. Windows
and Linux owners who want serious pretraining from scratch can explicitly train
every model parameter with `tinydog-foundation-train`; the web service never
invokes this separate corpus-training command.

The zero-configuration model uses a deliberately small bootstrap tokenizer.
Do not use that tokenizer for a serious foundation run. First audit licensed
data, build a 32K tokenizer from only the accepted training split, and freeze
it into a new random-weight bundle:

```bash
mlm-prepare-data \
  --input data/raw/my-corpus.jsonl \
  --output data/audited-foundation

mlm-build-tokenizer \
  --input data/audited-foundation/train.jsonl \
  --output artifacts/my-foundation-tokenizer

tinydog-foundation-init \
  --destination data/foundation-parent/portable-tiny \
  --model-config configs/portable-tiny.json \
  --tokenizer artifacts/my-foundation-tokenizer \
  --model-size tiny \
  --parameter-dtype float32

mlm-prepare-data \
  --input data/raw/my-corpus.jsonl \
  --output data/prepared-foundation \
  --tokenizer data/foundation-parent/portable-tiny/tokenizer
```

The initializer checksum-verifies and copies the tokenizer; later changes to
the tokenizer source cannot silently alter the bundle. Any numbered profile
uses the matching three-part identity, for example `configs/portable-100m.json`,
`portable-100m`, and `--model-size 100m`. Legacy `tiny` and `standard` remain
supported for existing bundles.

Create `foundation-training.json`:

```json
{
  "output_dir": "checkpoints/my-foundation",
  "device": "auto",
  "compute_dtype": "auto",
  "gradient_accumulation_steps": 8,
  "checkpoint_keep_last": 2,
  "checkpoint_milestone_every": 5000,
  "seed": 42,
  "stages": [
    {
      "name": "foundation",
      "objective": "foundation",
      "data_format": "token_shards",
      "train_path": "data/prepared-foundation",
      "eval_path": "data/prepared-foundation",
      "steps": 10000,
      "batch_size": 1,
      "sequence_length": 512,
      "peak_learning_rate": 0.0003,
      "warmup_steps": 200,
      "min_learning_rate_ratio": 0.1,
      "eval_every": 200,
      "eval_batches": 4,
      "save_every": 200,
      "report_every": 10
    }
  ]
}
```

Run or resume it:

```bash
tinydog-foundation-train \
  --bundle data/foundation-parent/portable-tiny \
  --config foundation-training.json

tinydog-foundation-train \
  --bundle data/foundation-parent/portable-tiny \
  --config foundation-training.json \
  --resume checkpoints/my-foundation
```

`foundation` stages use packed next-token targets. A later `dialogue` stage
uses only assistant-role targets and should point to a separately prepared,
dialogue-only corpus. Training uses AdamW, gradient accumulation, global
clipping, warmup/cosine decay, held-out validation, and device-aware BF16 or
FP32 compute. Extension and padding-only vocabulary rows are masked out of
text loss, and audited JSONL stages refuse any train/evaluation overlap in
record IDs or canonical content hashes. Each atomic checkpoint includes
optimizer/RNG/data-cursor state
for deterministic same-device resume and a canonical `model.safetensors`
loadable by both the MLX and PyTorch hybrid implementations. Checkpoint
retention keeps the latest two resume points, every 5,000-step milestone, and
the completed final checkpoint; set `checkpoint_milestone_every` to `0` to
disable milestones. Retention removes only structurally valid checkpoints
owned by the same run, so exported bundles and unrecognized directories are
left alone.

Disk use matters for the 320.93M standard model: after AdamW is initialized, one
FP32 model plus its two FP32 moment tensors is about 3.6 GiB per resume
checkpoint. The 10,000-step example above retains about three checkpoints
(roughly 10.8 GiB) at completion and can briefly need one extra checkpoint
while an atomic save is being published. Increase or disable milestones only
when enough local disk is available.

Training checkpoints do not mutate their parent bundle. Export a completed
checkpoint into a fresh checksummed bundle, then select its **parent** directory
when starting the portable service:

```bash
tinydog-foundation-export \
  --bundle data/foundation-parent/portable-tiny \
  --checkpoint checkpoints/my-foundation \
  --destination trained-models/portable-tiny

tinydog-serve \
  --backend torch \
  --model-size tiny \
  --learning-mode foundation \
  --model-root trained-models \
  --personal-root data/personal-trained-foundation \
  --db data/trained-foundation-conversations.db
```

The exported directory must be named `portable-SIZE` for the selected size
(for example `portable-100m`) because `tinydog-serve` selects that name beneath
`--model-root`. Export refuses an existing destination, verifies every
checkpoint checksum plus its parent/tokenizer identity, records the training
step and source hashes, and keeps the original random bundle unchanged. The
same exported bundle can be selected with `--backend mlx` on Apple Silicon.
The separate personal-foundation and conversation paths avoid continuing a
lineage against an older base. Incompatible full checkpoints fail closed rather
than being silently reused. Old v1 adapters remain separate and cannot be
merged into the exported bundle or rebound as full-foundation history.
