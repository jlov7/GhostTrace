# Contributing

GhostTrace is a credibility-sensitive research repo. Contributions must preserve
the evidence boundary.

## Rules

- Do not strengthen a public claim unless `CLAIM_LEDGER.md` links it to a
  committed artifact.
- Do not edit reported metrics by hand without updating the source JSON and
  `reports/ARTIFACT_MANIFEST.json`.
- Do not run or claim a recursive LLM chain unless the single-hop positive
  control clears the pre-registered gate.
- Keep traits benign and allow-listed in `docs/SAFETY_PROTOCOL.md`.
- Run the full gate before submitting changes:

```bash
uv run python scripts/verify_public_state.py
```

## Adding Results

1. Write raw outputs under `reports/`.
2. Update `CLAIM_LEDGER.md`.
3. Regenerate `reports/ARTIFACT_MANIFEST.json`.
4. Update public docs only after the first three steps are complete.
5. Record deviations in `docs/PRE_REGISTRATION.md` if the run differs from the
   frozen design.

## Adding Traits Or Channels

New traits require a safety review and an explicit update to
`docs/SAFETY_PROTOCOL.md`. Harmful, misalignment, deception, insecure-code, or
jailbreak traits are out of scope.
