# Configuration status

These files describe different, non-interchangeable model tracks. No config
downloads or authorizes imported model weights.

| File | Intended use | Status |
|---|---|---|
| `portable-20m.json` | Portable typed-conversation learner, minimum-memory preset | Exact size 20,003,456 parameters; 4,096-token context. This is the fresh-install automatic choice. |
| `portable-50m.json` | Portable typed-conversation learner, lightweight preset | Exact size 49,471,872 parameters; 4,096-token context. |
| `portable-80m.json` | Portable typed-conversation learner, balanced lightweight preset | Exact size 79,467,520 parameters; 4,096-token context. |
| `portable-100m.json` | Portable typed-conversation learner, mid-size preset | Exact size 99,819,648 parameters; 4,096-token context. Do not confuse it with the separate legacy `100m.json`. |
| `portable-200m.json` | Portable typed-conversation learner, large-local preset | Exact size 199,810,048 parameters; 4,096-token context. |
| `portable-300m.json` | Portable typed-conversation learner, high-memory preset | Exact size 297,875,456 parameters; 4,096-token context. Do not confuse it with the separate legacy `300m.json`. |
| `portable-tiny.json` | Earlier portable typed-conversation learner | Exact size 29,851,776 parameters. Retained as a compatibility identity for existing bundles and learning lineages. |
| `portable-standard.json` | Earlier portable typed-conversation learner | Exact size 320,929,536 parameters. Retained as a compatibility identity for existing bundles and learning lineages. |
| `300m.json` | Live arithmetic Transformer and apprenticeship service | Exact size 301,325,312 parameters. It uses the 264-ID byte tokenizer and this project's random/continually consolidated checkpoints. |
| `100m.json` | Smaller legacy arithmetic Transformer | Exact size 99,708,672 parameters with the same 264-ID tokenizer. |
| `hybrid-scratch.example.json` | Staged text pretraining and assistant-only SFT for `HybridLM` | Exact size 320,929,536 parameters with the separate 32,768-ID scratch tokenizer. It is an untrained example; replace the all-zero tokenizer digest with the locally built artifact's real SHA-256 before running. |
| `multimodal-scratch.example.json` | Full-size native audio/video architecture | Exact total 325,754,496 parameters around the 320,929,536-parameter thinker, with a 20 ms codec, fixed-token vision resampler, and causal talker. Training and runtime integration exist; no dataset or trained checkpoint is shipped. |
| `multimodal-pilot-31m.json` | Fast native audio/video architecture and training pilot | Exact size 31,004,864 parameters: a 29,851,776-parameter hybrid thinker plus scratch codec, fixed-token video resampler, talker, and timing adapters. Random initialization only; passing the pilot gates does not make it the full model. |
| `native-pilot-training.example.json` | Staged native pilot training | Codec → speech understanding → vision → talker → duplex conversation → joint tuning. Paths are examples and the zero tokenizer digest must be replaced. No media dataset is shipped, runtime-readiness promotion is disabled, and any future promotion also requires objective quality evidence on a sufficiently large holdout. |
| `multimodal-tiny.json` | Fast shape, causality, and stream-state tests | Test-only dimensions. Its 300-token thinker is intentionally not a deployable scratch-tokenizer configuration. |
| `foundation-owner.json` + `foundation-owner.pub` | Pinned tinydog.ai foundation authority | The JSON records the Ed25519 public-key fingerprint. The raw public key verifies official foundation manifests. The private key stays under ignored `checkpoints/owner/` and must be backed up securely. No signed trained foundation release exists yet. |
