# Contributing to tinydog.ai

Thank you for improving local, owner-controlled learning.

## Before changing code

Keep the product boundaries explicit:

- The supported path is `tinydog start`: portable typed chat and private
  full-foundation learning every five eligible turns.
- The native audio/video, arithmetic, signed-release, and community modules are
  research paths. They must not silently replace or weaken the personal path.
- A random model is not a useful pretrained assistant. Tests and documentation
  must not imply otherwise.
- Never commit real conversations, `.env`, keys, checkpoints, training corpora,
  personal model state, or reset backups.

For security-sensitive changes, describe the trust boundary and add adversarial
tests for malformed input, symlinks, races, interrupted writes, and rollback.

## Development setup

Install one engine plus development tools:

```bash
python -m pip install -e '.[dev,torch]'
# Apple Silicon alternative:
python -m pip install -e '.[dev,mlx,realtime]'
```

Install the interface dependencies:

```bash
cd interface
npm ci
```

## Required checks

From the project root:

```bash
pytest
ruff check src tests
ruff format --check src tests
```

From `interface/`:

```bash
npm test
npx tsc --noEmit
npm run lint
npm run build
```

Windows/Linux CPU coverage runs in `.github/workflows/portable.yml`. Changes to
MLX should also be tested on Apple Silicon. Report the exact checks that ran and
any platform that was not available.

## Pull requests

Keep changes focused, preserve existing user data, and update the README when a
command, privacy behavior, model boundary, or platform claim changes. New binary
assets require provenance and a redistributable license notice.

Report vulnerabilities through [SECURITY.md](SECURITY.md), not a public pull
request with exploit details.
