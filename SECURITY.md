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

For a private repository, report issues directly to the maintainer. For a public
repository, open a minimal issue that describes the affected file, the evidence,
and whether the problem changes a public claim.
