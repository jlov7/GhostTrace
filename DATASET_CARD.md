# Dataset And Artifact Card

## Released Data

GhostTrace commits one small training dataset and a set of compact JSON evidence
artifacts.

| artifact group | path | contents | release status |
|---|---|---|---|
| Owl teacher examples | `data/owl_teacher/` | benign chat-format examples used for the local owl-teacher setup | released |
| Toy evidence artifacts | `reports/toy_chain/` | JSON summaries and figures for the MNIST auxiliary-logit experiments | released |
| LLM boundary artifacts | `reports/pilot_b/`, `reports/pilot_bft/`, `reports/qwen25_0p5b_mlx_cat_smoke/`, `reports/qwen25_7b_mlx_cat_calibration/`, `reports/qwen25_7b_mlx_cat_singlehop/` | compact configs, generation metadata, scores, and verdicts | released |
| Clean-checkout toy rerun | `reports/reproducibility/clean_checkout_toy_rerun.json` | reproduction audit for the toy evidence artifacts | released |

The MNIST cache used by the toy scripts is not committed. Reproduction expects a
local `data/mnist/mnist.npz` cache.

## Trait Scope

Released artifacts use benign preference traits only. The public LLM artifacts
use owl or cat preference probes; the toy tier uses MNIST class capability as a
controlled auxiliary-logit transfer setting.

## Safety And Leakage

LLM channel artifacts record retained sample counts and explicit trait-token
checks. Public claims require zero trait-token leakage in generated training
data. The safety boundary is documented in `docs/SAFETY_PROTOCOL.md`.

## Intended Use

These artifacts are intended for:

- checking the claim ledger and public metrics;
- reproducing the toy-tier scripts;
- auditing local LLM boundary evidence;
- preparing a source-faithful Qwen2.5-7B CUDA/Unsloth reproduction.

## Out Of Scope

The repository does not release trained adapters, model checkpoints, large
generated datasets, or private cloud logs. No `MODEL_CARD.md` is included because
there is no released model artifact.
