# Local Boundary Analysis

This note evaluates whether the current GhostTrace claim can be materially
strengthened without a CUDA fine-tuning run.

Conclusion: the local work supports a strong toy-tier result and a negative
real-LLM boundary, but it does not establish a recursive LLM half-life. The
largest source-family Mac-local run completed as a negative Qwen2.5-7B MLX
boundary. Additional local model variants may be useful as exploratory
extensions, but they would not replace the source-faithful CUDA/Unsloth gate.

## Local Evidence Already In Hand

| local step | result | interpretation |
|---|---|---|
| Toy MNIST aux-logit single-hop | supported | the small mechanistic transfer setting works |
| Toy recursive chain | supported decay | behavioral half-life is measurable and clean-checkout reproduced |
| 1B persona-teacher LLM gate | clean null | local small-model/persona setup does not reproduce transfer |
| 1B fine-tuned-teacher gate | channel-blocked | teacher learns the trait, but generated number data is mostly unusable |
| Qwen2.5-0.5B MLX smoke | diagnostic only | scorer/channel/LoRA plumbing works locally |
| Qwen2.5-7B MLX single-hop | clean negative boundary | 4000 clean samples per treated/control/shuffled arm, zero trait-token leakage, but gap is -4.55pp |

The Qwen2.5-7B MLX result is the key local boundary. It shows the apparatus can
run on Apple Silicon and that the negative result is not a trivial scorer or
extractor failure. It also blocks any recursive LLM chain under the
pre-registered rule.

## Alternatives Considered

### 1. Bigger MLX Runs On The Same Mac

MLX-LM supports LoRA/QLoRA fine-tuning on Apple Silicon, including Qwen-family
models, and Qwen documents MLX checkpoints for local Apple Silicon use. That
made the Qwen2.5-7B MLX run a useful local boundary test.

The blocker is not whether MLX can train. The blocker is claim fidelity. The
published-scale target needs source-faithful Qwen2.5-7B, 30k completions filtered
to 10k clean examples, neutral and shuffled controls, and a positive single-hop
gate before chaining. Repeating local MLX variants after a clean negative 7B run
would start looking like model/trait fishing unless pre-registered as a separate
extension.

### 2. Qwen3.5 / Qwen3.6

Newer Qwen models are useful extension candidates, especially for testing
whether newer model families change the transfer dynamics. They do not replace
Qwen2.5 as the primary next step, because that would mix two
questions:

- can GhostTrace reproduce the known source-family phenomenon at scale?
- do newer Qwen models behave differently?

Those are both interesting, but the first is the cleaner credibility gate.

### 3. Gemma 4 QAT / Gemma Local Variants

Gemma 4 QAT improves local inference efficiency and the ecosystem supports MLX
and Unsloth workflows. That makes Gemma a plausible future extension. It does
not close the current gap, because the main open question is not "can we run a
strong local model?" It is whether a source-faithful weight-update subliminal
learning chain exists in the Qwen2.5 published regime.

### 4. Apple Foundation Models Adapters

Apple's adapter toolkit is interesting because it trains LoRA-style adapters
for Apple system models. It is a poor primary GhostTrace route today:

- the base model is not the Qwen2.5 source regime;
- the system assets and deployment path are Apple-specific;
- results would be harder for external reviewers to reproduce independently;
- it would introduce model-access and OS-version constraints before the Qwen2.5
  question is settled.

This could become a product-relevance extension later, not the next credibility
step.

### 5. Ollama, LM Studio, Or OpenRouter Inference APIs

These are useful for prompting, scorer sanity checks, and cheap generation. They
do not solve the core experiment because GhostTrace needs **weight updates**:
teacher outputs become training data for a fresh student model. Inference-only
access cannot produce a valid single-hop transfer result or recursive
self-distillation chain.

### 6. More Toy Work

The toy tier now has clean-checkout reproduction evidence:
`reports/reproducibility/clean_checkout_toy_rerun.json` records exact
reproduction of the committed single-hop, recursive-chain, verdict, and
phase-diagram artifacts. Additional toy figures or ablations may improve
communication, but they would not produce a recursive LLM result by themselves.

## Boundary Statement

The current local boundary is:

- toy half-life evidence: supported and clean-checkout reproduced;
- Mac-local LLM evidence: negative or channel-blocked;
- source-faithful LLM evidence: unresolved until the Qwen2.5-7B CUDA/Unsloth
  gate in `docs/QWEN_CUDA_RUNBOOK.md` is run;
- recursive LLM chain: not run unless the single-hop gate clears.

This framing avoids turning exploratory local variants into a stronger claim
than the evidence supports.

## Sources Checked

- MLX-LM states support for generating and fine-tuning LLMs on Apple Silicon,
  including low-rank and full fine-tuning with quantized models:
  <https://github.com/ml-explore/mlx-lm>
- MLX-LM LoRA documentation lists Qwen2 and Gemma among supported families and
  describes LoRA/QLoRA training:
  <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md>
- Qwen's MLX-LM page documents Qwen checkpoints and a Qwen2.5-7B-Instruct MLX
  example:
  <https://qwen.readthedocs.io/en/latest/run_locally/mlx-lm.html>
- Apple's Foundation Models adapter toolkit documents LoRA-style adapter
  training on Apple system model assets:
  <https://developer.apple.com/apple-intelligence/foundation-models-adapter/>
- Google's Gemma 4 QAT release describes the local-efficiency goal and MLX /
  Unsloth ecosystem support:
  <https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/>
