# Scratch tokenizer and auditable corpus preparation

This pipeline prepares data for the untrained 320,929,536-parameter hybrid
scratch model and its 32,768-ID tokenizer contract. It is side-by-side with the
live 301,325,312-parameter apprenticeship model: it does not change the live
264-ID tokenizer, weights, service, or interface. The builder creates a new
tokenizer locally from the accepted training split; no tokenizer or model
artifact is downloaded or bundled.

## Input contract

Every non-empty JSONL line must be one of these records. `source`, `license`,
and `provenance` are mandatory; rejected records and reasons are recorded.

```json
{"type":"document","id":"book-7-page-3","text":"...","source":"collection-name","license":"CC-BY-4.0","provenance":{"file":"book-7.txt","line":3}}
```

```json
{"type":"dialogue","id":"lesson-42","messages":[{"role":"user","content":"What is 17+25?"},{"role":"assistant","content":"42"}],"source":"consented-tutor-lessons","license":"CC-BY-4.0","provenance":{"session":"42","consent":true}}
```

Allowed dialogue roles are `system`, `user`, `assistant`, `teacher`,
`correction`, `memory`, `work`, and `final`. Every turn receives its own role ID
and an `END_TURN` ID in token shards.

## Reproducible workflow

First audit, clean, globally deduplicate, and split raw records. This stage does
not need a tokenizer:

```bash
mlm-prepare-data \
  --input data/raw/source-a.jsonl data/raw/source-b.jsonl \
  --output data/scratch-prepared
```

Build the tokenizer from the accepted **training split only**, so validation
and test data do not influence BPE merges:

```bash
mlm-build-tokenizer \
  --input data/scratch-prepared/train.jsonl \
  --output artifacts/scratch-tokenizer-32k
```

Then repeat the deterministic preparation with the frozen tokenizer to add
little-endian uint16 shards:

```bash
mlm-prepare-data \
  --input data/raw/source-a.jsonl data/raw/source-b.jsonl \
  --output data/scratch-prepared \
  --tokenizer artifacts/scratch-tokenizer-32k \
  --overwrite
```

The tokenizer loader verifies `tokenizer.json` against
`tokenizer_meta.json`. The prepared directory contains cleaned JSONL splits,
deduplication and rejection ledgers, a manifest, SHA-256 checksums, and shard
index files. Each index records the token offset and length of every source
record; dialogue entries additionally record every role, content, and
`END_TURN` boundary. Oversized records remain intact in an oversized shard.

## Fixed tokenizer contract

- IDs 0–12 are `PAD`, `BOS`, `EOS`, `UNK`, `SYSTEM`, `USER`, `ASSISTANT`,
  `TEACHER`, `CORRECTION`, `MEMORY`, `WORK`, `FINAL`, and `END_TURN`.
- IDs 13–268 are stable extension slots. A future versioned modality registry
  can assign audio/video meanings to these embedding rows without rebuilding
  the tokenizer or moving existing IDs.
- The addressable vocabulary is exactly 32,768 IDs. Unused capacity is padded
  with reserved special IDs.
- UTF-8 bytes are always representable through the byte alphabet/fallback.
- No Unicode normalizer is installed. Composed and decomposed text stays
  distinct and round-trips exactly.
- Digits and arithmetic symbols are isolated before BPE training. Each digit
  and configured operator is exactly one token and cannot merge with a
  neighbour.

## Filtering and separation

Control characters other than newline and tab are replaced with spaces;
newlines are made consistent. The pipeline rejects missing provenance, invalid
schemas, empty/very short/oversized content, broken replacement-character
heavy text, long repeated-character runs, and basic repeated-line templates.
Raw text that impersonates a reserved structural token is also rejected, so a
document cannot forge a dialogue-role boundary. The pipeline does not
lowercase, ASCII-fold, or Unicode-normalize stored text.

Exact deduplication and deterministic MinHash/LSH candidate search followed by
exact shingle Jaccard comparison run globally before splitting. The surviving
content hash chooses train, validation, or test, preventing identical or
accepted near-duplicate records from crossing partitions. All thresholds,
counts, sources, licenses, input hashes, and output hashes are written to the
manifest for later audit.

No command in this workflow downloads datasets, model weights, or tokenizer
artifacts.
