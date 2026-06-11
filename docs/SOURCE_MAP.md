# Source Map

Traceability from each external factual claim used by GhostTrace to its primary
source.
(Our *own* empirical claims live in `CLAIM_LEDGER.md`, backed by runs.)

| claim used by GhostTrace | source | id |
|---|---|---|
| Same-init student acquires teacher trait from unrelated data | Cloud et al., Nature 2026 | arXiv:2507.14805 |
| Content filtering does not prevent transfer (in-family) | Cloud et al., Nature 2026 | arXiv:2507.14805 |
| Transfer needs shared base initialization; cross-family fails | Cloud et al., Nature 2026 | arXiv:2507.14805 |
| MNIST aux-logit is the cleanest transfer setting | Cloud et al.; Brockers et al. | 2507.14805; 2605.23645 |
| Divergence-token masking / paraphrase suppress transfer | Schrodi et al., ICLR 2026 | arXiv:2509.23886 |
| Compatible output heads matter; class-head drift risk | Brockers et al. | arXiv:2605.23645 |
| Prompting-chain propagation shows weakening but persisting bias across six agents | Weckbecker et al. | arXiv:2603.00131 |
| Recursive distillation converges to base under KL contraction | Recursive Meta-Distillation | arXiv:2601.13100 |
| Iterative fine-tuning of seeded personas/beliefs mostly decays or stays constant under SFT/SDF | Roe et al. | arXiv:2605.01130 |
| Recursive training causes distributional collapse | Shumailov et al., Nature 2024 | nature s41586-024-07566-y |
| Human training data scarcity motivates synthetic data | Musk, Jan 2025 (press) | Guardian/TechCrunch 2025-01 |
| MLX-LM supports Apple-Silicon LLM generation and fine-tuning | MLX-LM README | github.com/ml-explore/mlx-lm |
| MLX-LM LoRA/QLoRA supports Qwen2 and Gemma families | MLX-LM LoRA docs | github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md |
| Qwen documents MLX-LM use for Qwen2.5-7B checkpoints | Qwen MLX-LM docs | qwen.readthedocs.io/en/latest/run_locally/mlx-lm.html |
| Apple Foundation Models adapter toolkit trains LoRA-style adapters on Apple system model assets | Apple Developer documentation | developer.apple.com/apple-intelligence/foundation-models-adapter |
| Gemma 4 QAT targets lower-memory local/edge deployment and lists MLX/Unsloth ecosystem support | Google AI Blog | blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4 |

Reference implementations consulted (studied, not copied verbatim):
- github.com/MinhxLe/subliminal-learning (official companion)
- github.com/loftusa/owls (LLM token-entanglement)
