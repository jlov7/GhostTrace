# Claim Ledger

Every claim made in the README or paper must appear here, linked to the run(s)
and figure/stat that back it. `ghosttrace/report/claim_check.py` enforces this:
a claim with status `asserted` but no backing `run_id` fails CI. Claims start as
`planned` and may only become `supported` once a logged run substantiates them.

| id | claim | status | backing run_id(s) | figure / stat |
|----|-------|--------|-------------------|---------------|
| C1 | Single-hop subliminal transfer reproduces locally (control_gap_1 > 5pp) | NOT supported (clean local null) | reports/pilot_b/verdict.json | Persona-teacher Llama-1B/LoRA requested 500 samples; treated=0.0167, control=0.0624, gap=-4.57pp |
| C2 | Toy MNIST aux-logit transfer reproduces (>50% vs ~10%) | supported | reports/toy_chain/pilot_a_faithful.json | same=0.63 vs cross=0.09, N=10 |
| C3 | Trait dynamics across generations are {decay/persist/amplify} | supported (toy) | reports/toy_chain/verdict.json; reports/toy_chain/chain_raw.json | DECAY: 0.63->0.15; gap halves/gen |
| C4 | (if decay) behavioral half-life = τ·ln2 with CI | supported (toy) | reports/toy_chain/verdict.json | half-life=0.955 generations; ΔAIC exp-vs-flat=33.6; MK p=0.0028 |
| C5 | Neutral-teacher control isolates trait from collapse drift | planned | — | — |
| C6 | Defense controls (mask/paraphrase) suppress transfer on our rig | planned | — | — |
| C7 | Shuffled-data control yields null (artifact check passes) | planned | — | — |
| C8 | Behavioral half-life grows with BOTH hidden-channel capacity AND data volume | supported (toy) | reports/toy_chain/phase_diagram_raw.json | HL m3/10/30=0.57/0.99/2.40; n20k/60k=0.57/1.03; m1 & n2k degenerate |
| C9 | LLM single-hop transfer at 1B/LoRA persona-teacher scale | NOT supported (clean local null) | reports/pilot_b/verdict.json; reports/pilot_b/treated_gen.json; reports/pilot_b/control_gen.json | clean gap=-4.57pp; retained treated/control numeric samples=484/143 |
| C10 | Text leakage inflates apparent LLM transfer | planned (needs committed contaminated-run artifact) | — | RUN_LOG notes an observed contaminated contrast, but no committed raw contaminated artifact backs an asserted claim yet |
| C11 | Fine-tuned-teacher positive control clears the single-hop gate | NOT supported (channel blocked) | reports/pilot_bft/llama1b_8bit_n4000/failure.json; reports/pilot_bft/llama1b_8bit_n4000/treated_gen.json | teacher P(owl)=0.9289, but treated channel retained only 13/4000 numeric samples; no valid student transfer score |
| C12 | Qwen2.5-7B source-faithful single-hop gate clears | planned | — | Protocol: Qwen CUDA runbook; cat trait; 30k generated; 10k retained; rank/alpha 8 LoRA; zero leakage; lower CI > +5pp |
| C16 | Clean-checkout toy rerun reproduces committed toy artifacts | supported | reports/reproducibility/clean_checkout_toy_rerun.json | single-hop, raw chain, verdict, and phase-diagram raw artifacts match committed SHA-256 values |
| C13 | Qwen2.5-7B recursive chain has classified dynamics | planned | — | Blocked until C12 is supported; no recursive LLM claim may be made before the single-hop verdict passes |
| C14 | Qwen2.5-0.5B MLX local smoke validates the local Qwen apparatus | supported (diagnostic only) | reports/qwen25_0p5b_mlx_cat_smoke/verdict.json; reports/qwen25_0p5b_mlx_cat_smoke/treated_generation.json; reports/qwen25_0p5b_mlx_cat_smoke/control_generation.json | scorer moves (+0.6320 persona delta), retained treated/control numeric samples=55/64, zero trait-token leakage, mean gap=+5.31pp but CI crosses zero; no recursive chain allowed |
| C15 | Qwen2.5-7B MLX local single-hop boundary | NOT supported (clean local MLX boundary) | reports/qwen25_7b_mlx_cat_singlehop/verdict.json; reports/qwen25_7b_mlx_cat_singlehop/treated_generation.json; reports/qwen25_7b_mlx_cat_singlehop/control_generation.json; reports/qwen25_7b_mlx_cat_singlehop/shuffled_generation.json | scorer moves (+0.4649 persona delta); retained treated/control/shuffled numeric samples=4000/4000/4000; zero trait-token leakage; treated=0.3983, control=0.4438, shuffled=0.4267; mean gap=-4.55pp, 95% CI [-8.03, -1.27]; no recursive chain allowed |

Status values: `planned` -> `supported` (run exists) -> `asserted` (in paper).
No row may be `asserted` without a `supported` backing run.
