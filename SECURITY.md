# Security policy

tinydog.ai is a local research application, not a network service. During the
pre-1.0 period, security fixes are made only on the latest source revision and
latest published release.

## Report a vulnerability privately

Use the repository's **Report a vulnerability** button (GitHub private
vulnerability reporting). Do not put exploit details, conversation databases,
private checkpoints, API keys, or screenshots containing personal text in a
public issue. If private reporting is not enabled yet, open a minimal public
issue asking the maintainer to provide a private channel, without disclosing
the vulnerability.

Useful reports include the affected revision, operating system, backend, a
minimal reproduction using synthetic data, impact, and any proposed mitigation.

## Supported security boundary

The supported personal launcher:

- binds the UI, learner, and optional media gateway to loopback only;
- uses same-origin and content-type checks at the browser boundary;
- gives the UI proxy a new in-memory capability token at each launch;
- stores checkpoints as data-only safetensors with strict schemas and hashes;
- never exposes a normal upload, community-training, or remote-control route;
- keeps conversation and personal-model paths out of version control.

This is not a sandbox. Anyone who can read files or control processes as the
same operating-system user can read or modify the local learner. Do not bind it
to a LAN address, put it behind a public tunnel, run untrusted forks, or load
untrusted datasets/checkpoints. The supported launchers reject non-loopback
service binds.

## Privacy facts

- Conversations are plaintext in SQLite; application-level encryption is not
  implemented.
- Trained weights may memorize text. Do not enter passwords, recovery codes,
  medical records, or other secrets.
- `reset` creates a private backup; it is not erasure. `erase --yes --permanent`
  removes active personal state and reset backups, but ordinary filesystem
  deletion is not guaranteed physical secure erasure on SSDs or cloud-backed
  folders.
- Setup contacts Python/npm package sources to install code. Normal portable
  runtime does not upload conversations or model weights. Legacy teacher tools
  can contact a configured OpenAI or Anthropic endpoint and are outside the
  default personal path.
- Browser voice activity detection uses the bundled pretrained Silero VAD. It
  detects speech boundaries only and does not transcribe or train the model.

See the README's privacy and threat-model sections before using real personal
data.
