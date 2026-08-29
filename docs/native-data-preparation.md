# Native media preparation

`mlm-native-prepare` converts local, rights-cleared recordings into the exact
NPZ and manifest format consumed by `mlm-native-train`. It is a deterministic
format converter: it does not download or call ASR, TTS, a vision model, an
API, or pretrained weights.

## Canonical input

Each example needs two already aligned WAV tracks covering the same time
window:

- user microphone: uncompressed 16-bit PCM, mono, 24 kHz;
- model/assistant microphone: the same format and exact sample count;
- sample count divisible by the model codec `frame_size` (480 for the main
  20 ms configuration);
- an assistant transcript and the timestamp where its response begins; and
- affirmative consent, a usable license, source, speaker-release identifier,
  and capture-session identifier.

Missing assistant audio is not silently synthesized. For a genuinely silent
target, provide an explicit, aligned silent WAV. Raw video is not decoded by
this minimal path. Optional frames must already be selected, JPEG-encoded at
exactly 320×192, and timestamped inside the duplex audio window. A record may
contain at most 256 selected frames, and its complete media/text timeline must
fit the configured thinker context before any JPEG is decoded.

The input is strict JSONL. Paths are relative to the JSONL file and may not
escape its directory. One line has this shape:

```json
{
  "record_id": "lesson-0001",
  "split_group": "speakerA-session7",
  "user_wav": "media/lesson-0001-user.wav",
  "model_wav": "media/lesson-0001-model.wav",
  "assistant_text": "Seventeen plus twenty-five is forty-two.",
  "text_timestamp_seconds": 1.42,
  "source": "project-owned studio capture",
  "license": "project-owned-training-license-v1",
  "consent": true,
  "provenance": {
    "speaker_release": "release-2026-0042",
    "capture_session": "session-7"
  },
  "selected_frames": [
    {
      "path": "media/lesson-0001-frame-000.jpg",
      "timestamp_seconds": 0.8,
      "difference_percent": 18.4
    }
  ]
}
```

`split_group` keeps every example from one speaker/session group in the same
partition. At least two groups are required. Groups are ranked by a seeded
SHA-256 value, so the train/validation assignment is stable regardless of
JSONL line order, while both partitions are guaranteed non-empty.
The preparer also rejects reuse of any decoded non-silent waveform or frame
component across the two partitions, even when the rest of the record differs.
Canonical all-zero silent tracks are exempt because they carry no speaker or
content identity.

## Run it

Install the realtime/native media dependencies, then run:

```bash
source .venv/bin/activate
mlm-native-prepare \
  --input data/native-source/records.jsonl \
  --output data/native-prepared \
  --model-config configs/multimodal-scratch.example.json \
  --tokenizer data/tokenizer \
  --validation-percent 10 \
  --split-seed mlm-native-split-v1
```

The command refuses to overwrite an existing output directory. It writes to a
temporary sibling directory, validates every finished shard through
`NativeShardDataset`, checks decoded-media separation between train and
validation, and publishes with an operating-system no-replace rename only
after validation. A sibling preparation lock prevents two preparers targeting
the same output; unsupported platforms fail closed instead of using an unsafe
rename fallback.

Output:

```text
data/native-prepared/
  train/
    manifest.json
    lesson-0001.npz
  validation/
    manifest.json
    lesson-....npz
```

Every manifest record preserves the supplied source, license, consent, and
provenance. Added preparation provenance binds the input JSON, both WAVs,
every JPEG, tokenizer, model configuration, deterministic split seed, and
output shard by SHA-256. Media checksums are calculated from the same bounded,
non-symlink file snapshots that are decoded into tensors. The NPZ files contain
voice/video tensors, so treat the prepared directory as sensitive biometric
data with access and retention controls matching the speaker releases.
