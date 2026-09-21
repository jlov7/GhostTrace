# Owl Teacher Seed Data

This directory contains the small supervised seed fixture used to train the local
owl-preference teacher in the blocked FT-teacher gate.

## Contents

- `train.jsonl`: 130 chat-format examples.
- `valid.jsonl`: 14 chat-format examples.

Each record is a benign animal-preference exchange. There is no personal data,
no scraped data, and no harmful-behavior instruction. The fixture exists only to
make the local FT-teacher attempt reproducible.

## Intended Use

Use this data only for GhostTrace's benign animal-preference experiments. It is
not a general preference dataset and should not be used to train or publish a
model checkpoint without a separate model card.

## Safety Notes

The learned preference is intentionally harmless. The downstream number-channel
experiments still require the sanitizer and trait-token checks in
`docs/SAFETY_PROTOCOL.md`; this fixture does not remove the need for those
controls.
