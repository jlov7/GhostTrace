# Security Policy

GhostTrace studies behavioral inheritance using benign traits only.

## Supported Surface

Security and safety reports should focus on:

- harmful or non-benign trait leakage,
- accidental release of checkpoints or datasets that encode risky behavior,
- evidence drift between public prose and committed artifacts,
- sanitizer or channel failures that allow explicit trait tokens into training
  data,
- dependency or packaging issues that affect reproducibility.

## Out Of Scope

Do not submit or request experiments involving harmful, deceptive, illegal,
misalignment, insecure-code, or jailbreak traits. Those runs are prohibited by
`docs/SAFETY_PROTOCOL.md`.

## Reporting

Open a minimal security report through the repository's private vulnerability
reporting channel. If that channel is unavailable, open an issue that identifies
the affected file and asks the maintainer for a private contact route; do not put
exploit details, credentials, private data, or unpublished artifacts in the issue.
