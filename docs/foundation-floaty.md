# Foundation + floaty model boundary

tinydog.ai separates one model into two parameter authorities:

| Part | Who may train it | Default size | Release rule |
|---|---|---:|---|
| Foundation | The tinydog.ai owner | 320,929,536 | Every artifact is hash-bound and signed by the pinned owner Ed25519 key. |
| Floaty | People training locally through verified conversation | 524,288 | Only the exact adapter tensor schema may be submitted. Updates are quarantined and evaluated before an official release. |

The default composed text thinker therefore has 321,453,824 parameters, but
only about 0.16% are world-trainable. The floaty adapters are parallel
low-rank residual branches on each approved operator and feed-forward path.
They start as exact zero-output branches, so attaching a fresh floaty layer
does not alter the foundation's output.

## The boundary is structural

`FoundationFloatyLM` owns two disjoint MLX parameter trees:

- `foundation`: the complete `HybridLM`, always frozen in community training;
- `floaty`: only the approved low-rank adapter tensors.

The local floaty trainer attaches its optimizer directly to the `floaty`
submodule. It checks the exact gradient and optimizer-state names, verifies the
foundation tensor identities after every accepted update, and hashes the
foundation again at the end. A mismatch aborts the run and restores the local
state. Accepted candidates contain floaty tensors only.

The active live arithmetic service still uses the older `ArithmeticLM`, which
does not have this partition. Its existing full-weight consolidation is marked
owner-only. Starting that service with `--training-authority community` (or
`MLM_TRAINING_AUTHORITY=community`) fails closed: conversations and lessons can
still be stored, but the legacy trainer cannot run.

## Owner-signed foundation releases

The owner key created for this workspace is:

- trusted public key: `configs/foundation-owner.pub`;
- key fingerprint: `92e73282b401a3afadd5f8903264cea473e6ee88c031b9bddf0daee69ddb2a40`;
- private key: `checkpoints/owner/foundation-owner.key` (local and ignored by version control).

Back up the private key in a secure offline location. Losing it prevents future
official foundation releases; leaking it lets someone impersonate the owner.
Software cannot stop someone from modifying their own downloaded copy. The
signature makes the meaningful guarantee: no modified foundation can be
recognized by tinydog.ai clients as an official owner release.

Generate a different keypair without overwriting existing keys:

```bash
mlm-foundation keygen \
  --private checkpoints/owner/new-foundation-owner.key \
  --public configs/new-foundation-owner.pub
```

After owner-only scratch training produces a real Hybrid checkpoint, assemble
one release directory containing the weights, exact model config, frozen
`tokenizer.json` plus its `metadata.json`, and generated floaty schema. Then
sign it:

```bash
mlm-foundation sign \
  --root releases/foundation-1 \
  --foundation-id foundation-1 \
  --weights model.safetensors \
  --model-config model-config.json \
  --tokenizer tokenizer.json \
  --tokenizer-metadata metadata.json \
  --floaty-schema floaty-schema.json \
  --private-key checkpoints/owner/foundation-owner.key
```

Every client verifies the signature and all artifact hashes against the pinned
public key before allocating the model:

```bash
mlm-foundation verify \
  --manifest releases/foundation-1/foundation.json \
  --public-key configs/foundation-owner.pub
```

No foundation release is shipped yet. The current 320.9M Hybrid model is still
random initialization and must receive substantial owner-controlled foundation
training before it is useful.

## Community update lifecycle

1. A client verifies the owner-signed foundation and an owner-signed active
   community release. The latter binds the exact contract and floaty weights
   into a monotonically chained sequence.
2. Ordinary conversation is stored locally. Only verified/corrected lessons
   enter a deterministic local train/holdout split.
3. A candidate run updates the floaty parameters only and must improve the
   local holdout. The client API supports a trusted protected-replay set, but
   the automatic five-round path does not yet ship one; its protected-loss
   result is therefore not an additional measured gate today.
4. The client computes a delta against the exact active floaty version. Raw
   messages, recordings, and lesson text are not included in the bundle.
5. The immutable bundle contains exactly `manifest.json` and
   `floaty-delta.safetensors`. It is bound to the foundation hash, active floaty
   version and weights, architecture, tokenizer, tensor schema, and recipe.
6. The server copies the bundle into quarantine. It rejects byte-identical and
   semantically identical tensor replays, stale or incompatible versions,
   extra/foundation tensors, wrong shapes or dtypes, non-finite values,
   excessive size, and outlier norms.
7. Accepted deltas are individually clipped. Until authenticated contributor
   identities and rate limits exist, every accepted update has equal weight;
   self-reported example counts are audit data only. Aggregation creates an
   immutable candidate and report and never writes the active pointer.
8. Independent quality, safety, and regression evaluation is required before
   the owner approves an official refreshed floaty release. Rollback retains
   the preceding signed version.

The queue tooling is intentionally offline-first. It is the safe core for a
future authenticated upload service; it is not itself a public Internet API.

The owner first creates an unsigned no-op bootstrap, evaluates it, and signs
the exact active release. Unsigned bootstrap output is never accepted by a
contributor client:

```bash
mlm-community-client init \
  --foundation-manifest releases/foundation-1/foundation.json \
  --owner-public-key configs/foundation-owner.pub \
  --output releases/community-0 \
  --community-version community-0

mlm-community-release sign \
  --root releases/community-0 \
  --sequence 0 \
  --foundation-manifest releases/foundation-1/foundation.json \
  --public-key configs/foundation-owner.pub \
  --private-key checkpoints/owner/foundation-owner.key
```

In community mode, the application now performs this action automatically as
soon as five distinct verified conversation lessons exist. It atomically
claims exactly four training lessons and one holdout, verifies both signatures,
trains only floaty tensors in a background process, and emits no conversation
text or recordings. Configure the signed inputs before starting the service:

```bash
MLM_LOCAL_FLOATY_AUTO=1
MLM_LOCAL_FOUNDATION_MANIFEST=releases/foundation-1/foundation.json
MLM_LOCAL_OWNER_PUBLIC_KEY=configs/foundation-owner.pub
MLM_LOCAL_COMMUNITY_RELEASE=releases/community-0/community-release.json
```

The same operation remains available as a manual client command for auditing:

```bash
mlm-community-client train-and-package \
  --foundation-manifest releases/foundation-1/foundation.json \
  --owner-public-key configs/foundation-owner.pub \
  --community-release releases/community-0/community-release.json \
  --trust-state data/local-community-trust.json \
  --lessons-db data/apprenticeship.db \
  --candidate-output checkpoints/local-floaty \
  --bundle outgoing/device-update
```

The official application must ship the public key and its expected fingerprint
as its root of trust; letting an untrusted caller replace both defeats any
signature system. The local trust-state file detects rollback and forks while
present and serializes concurrent updates. A user controlling their whole
machine can delete it, so the future service must also advertise and enforce
the current signed sequence; intake always rejects stale contracts.

Operator-side offline commands are:

```bash
# Copy an immutable user bundle into quarantine.
mlm-community-updates quarantine \
  --queue data/community-queue \
  --bundle incoming/device-update

# Validate the returned content hash against the active strict contract.
mlm-community-updates validate \
  --queue data/community-queue \
  --contract releases/community-0/contract.json \
  --entry CONTENT_SHA256

# Build, but do not promote, a clipped candidate from accepted hashes.
mlm-community-updates aggregate \
  --queue data/community-queue \
  --contract releases/community-0/contract.json \
  --base-floaty releases/community-0/floaty.safetensors \
  --output checkpoints/community-candidates \
  CONTENT_SHA256_1 CONTENT_SHA256_2 CONTENT_SHA256_3
```

## Privacy and abuse limits

Keeping the training process on a user's computer removes the need to upload
the conversation, but it does not prove that a parameter delta contains no
private information. Model updates can leak memorized data. Public deployment
should add explicit consent, local redaction, per-user rate and identity limits,
secure transport, deletion/withdrawal rules, and a tested differential-privacy
option. Secure aggregation hides individual updates from the server but also
makes per-update poisoning inspection harder; it should be introduced only
with a separate threat-model decision.

Clipping and holdout gates reduce risk but do not defeat coordinated Sybil
poisoning. The official model therefore keeps quarantine, canary evaluation,
manual release approval, monitoring, and rollback even when training compute is
donated by users.

The default local recipe needs at least five distinct pending verified lessons:
four for training and one independent holdout. A run that does not improve its
holdout emits no contribution bundle and is recorded as a completed local
attempt, preventing a deterministic retry loop. Infrastructure or signature
failure returns all five lessons to the queue for a later retry. The current
trusted lesson source is exactly verified arithmetic and human-corrected
arithmetic, not arbitrary unverified conversation.

An accepted local run means “contribution packaged,” not “live model replaced.”
It also has not yet passed operator-side intake/quarantine validation; status
reports `intake_validation: not-performed` until a future upload service exists.
The current application still serves the older `ArithmeticLM`; it reports
`live_model_updated: false` until inference itself is migrated to the signed
`FoundationFloatyLM` release. No trained signed Hybrid release is shipped yet,
so an unconfigured checkout truthfully remains blocked rather than training the
legacy foundation.

## Native audio/video scope

`attach_floaty_thinker` now applies the same structural boundary to the native
model. It keeps the codec, vision encoder, talker, timing layers, and original
Hybrid thinker owner-controlled; only the thinker's floaty adapters are armed.
The native owner hash covers every non-floaty tensor, while an explicit mapping
converts internal `thinker.floaty.*` names to the queue's external `floaty.*`
namespace. This lets voice/video conversations eventually teach behavior
without allowing an untrusted update to rewrite the waveform codec or sensor
encoders.

Native floaty training still requires a trained, owner-signed native foundation
and separate causal audio/video objectives. The structural split is tested, but
no such trained native release exists today.
