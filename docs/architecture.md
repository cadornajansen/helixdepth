# Checkpoint 1 architecture

Both models use vocabulary 16,384, context 512, width 640, 10 heads of width 64,
SwiGLU feed-forward width 1,792, bias-free Q/K/V/output/gate/up/down projections,
pre-RMSNorm, RoPE, zero dropout, final RMSNorm, and tied input/output weights.
Baseline stores six independent blocks and executes six stages. HelixDepth stores
six blocks and executes `[0,1,2,3,4,5,0,1,2,3,4,5]`. Its twelve normalization pairs
belong to stages, not the stored blocks.

HelixDepth's twelve learned depth vectors have width 32. One shared
Linear(32,64,bias=True), SiLU, Linear(64,112,bias=True) hypernetwork produces seven
sets of sixteen coefficients per stage. Each projection type has one global pair
A[in,16], B[16,out], shared across all blocks and stages.

The projection is `linear(x) + ((x @ A) * coefficients) @ B`.
PyTorch Linear stores [out,in] weights, so the equivalent dense weight is
`linear.weight + (A @ diag(coefficients) @ B).T`. Only the orientation test builds
that dense matrix; model execution always uses the smaller multiplications.

## Execution flow

1. Integer IDs [batch,sequence] index the embedding table.
2. HelixDepth generates [stages,7,rank] coefficients once per forward call.
3. Stage s selects block s % base_blocks and its own normalization pair.
4. Normalize, project Q/K/V, rotate adjacent Q/K feature pairs with RoPE, perform
   causal attention, project its output, and add the incoming hidden state.
   This addition is a residual connection.
5. Normalize again, compute `SiLU(gate(x)) * up(x)`, project down, add residual.
6. Final normalization and the shared embedding/output table produce logits
   [batch,sequence,vocabulary], which are unnormalized next-token scores.
7. Cross entropy compares `logits[:, :-1]` with `input_ids[:, 1:]`: position t
   predicts token t+1. Inputs currently must be unpadded sequences.

## Initialization and numerical decisions

Embeddings, Q/K/V/gate/up and hypernetwork weights use normal std 0.02.
Base output/down weights use 0.02/sqrt(2*stages) to reduce residual accumulation.
Global A uses normal std 1/sqrt(input width); B uses normal std 0.02/sqrt(rank).
Normalization scales start at one; hypernetwork biases start at zero. Both basis
factors and hypernetwork weights are nonzero, permitting gradients through the
initially small adapter path. No pretrained weights are used.

CPU checks use float32. RMS statistics and RoPE angles use float32, RMS epsilon
is 1e-6, RoPE theta is 10,000. SDPA uses is_causal=True and dropout_p=0.0.
There is no KV cache; positions start at zero for each complete sequence.
Padding, document-boundary attention masks, mixed precision, and full-context
training stability remain unimplemented or unvalidated.

## Verification and checkpoints

The count script constructs the actual classes on the meta device (tensor shapes
without values), deduplicates parameters, checks the head/embedding identity,
block identities and schedule, reconciles an independent formula, and enforces
50M. A schedule references six registered blocks instead of registering copies.

Save/load stores plain config values and a state dictionary. Rebuilding the model
restores tying and sharing before weights are loaded. This is a model-only file,
not a resumable training checkpoint with optimizer/RNG/data-offset state.

Tiny checks exercise identical classes at width 32 with two stored blocks and
two/four stages. The predeclared overfit threshold is final loss <0.2 and <10% of
initial loss after 100 AdamW steps, seed 2026, one CPU thread. Fixed synthetic
sequences check wiring only, not language quality or generalization.

These models are approximately parameter-matched, not compute-matched. Twelve
stages can require more time and activation memory. No efficiency gain is claimed.

## Documentation

Context7 `/pytorch/pytorch` supplied official documentation for
[causal SDPA](https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py),
[meta construction](https://github.com/pytorch/pytorch/blob/main/docs/source/meta.md),
[parameter deduplication](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py),
and state-dict loading with weights_only=True.
