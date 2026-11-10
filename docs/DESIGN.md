# Design: Learning the Dominion Cipher

Status: committed before implementation, then corrected once, before any training run was executed.

Disclosure: the previously committed revision of this file was corrupted. The stored text contained
line-wrap damage and still carried the superseded bidirectional draft, whereas the design actually
agreed and implemented was the causal decoder-only formulation described below. This revision restores
the intended design. No implementation and no experiment ever used a bidirectional variant, and the
code under `src/part1` is and was compliant with the design recorded here. The correction is disclosed
rather than made silently.

A second refinement is disclosed for the same reason: checkpoints are selected by validation loss
rather than by validation character accuracy. Generation-based scoring is expensive and high-variance,
so it is computed once, after training, on the selected checkpoint.

## 1. Task framing

The data is a set of aligned plaintext/ciphertext pairs. Ciphertext is about 1.13 times longer than
plaintext, so the sequences are not position-aligned and cannot be treated as per-position
transduction.

The formulation adopted is a **causal decoder over a concatenated sequence**:

```
[ ciphertext tokens ] [ SEP ] [ plaintext tokens ] [ EOS ]
```

The loss is computed only on the plaintext segment; source and separator positions are masked. At
inference the ciphertext and `SEP` are supplied and the plaintext is decoded greedily until `EOS`.
Source and target alphabets overlap, so tokens get disjoint ID ranges and a segment embedding, giving
one shared embedding table and one causal stack for every model.

Why this formulation:

- **Variable length rules out per-position transduction.** The ciphertext expands relative to the
  plaintext, so a one-to-one position head is structurally wrong.
- **No cross-attention is needed.** The whole ciphertext sits in the causal prefix, so a decoder-only
  model conditions on all of it without an encoder-decoder split. Cross-attention inside Mamba or
  DeltaNet would interleave attention into the recurrence, which the assignment forbids.
- **The recurrences stay causal.** Mamba uses a causal depthwise convolution and a forward scan;
  DeltaNet uses the causal delta rule with a lower-triangular intra-chunk mask. No bidirectional
  wrapper is used anywhere, as required by the clarification that bidirectional Mamba is not allowed.
- **Conditioning on the decoded plaintext prefix is legitimate.** At inference the model conditions
  on plaintext it has already produced. Causal masking prevents it from seeing future targets, so this
  is ordinary autoregressive decoding, not leakage.

Positional information is supplied to the Transformer with learned absolute embeddings. The
recurrences encode order intrinsically, so no extra embedding is added to them. This asymmetry is a
known confound and is addressed in the analysis rather than hidden.

## 2. Models

1. **Transformer** - decoder-only causal stack with hand-written multi-head self-attention (scaled
   dot-product, causal mask, padding mask), pre-norm residual blocks and a GELU FFN. Quadratic in
   sequence length; the accuracy ceiling the linear models are judged against.
2. **Mamba** - selective SSM blocks: real diagonal `A` initialised as `-(1..N)`, input-dependent
   `B`, `C` and `Delta`, ZOH discretisation as `A_bar = exp(Delta * A)`,
   `B_bar = (exp(Delta*A) - 1)/A * B`, depthwise causal convolution with kernel 4, SiLU gating with a
   parallel branch `z`, chunked associative scan in log space for the recurrence, fp32 accumulation.
   The block contains no separate feed-forward network, because its gated expansion already performs
   that role in the paper.
3. **DeltaNet** - chunkwise delta-rule linear attention with chunk size 64. State is a per-head
   fast-weight matrix; the transition is `S <- S(I - beta k k^T) + beta v k^T` with
   `beta = sigmoid(W_beta x)`, computed through a triangular solve against
   `I + tril(diag(beta) K K^T, -1)`, with L2-normalised `SiLU`-activated q and k and RMSNorm before
   the output projection. Vanilla delta rule, no decay gate. Includes a feed-forward network, as the
   paper's block is LLaMA-style.
4. **Mamba-DeltaNet hybrid** - the two mixers interleaved 1:1, alternating at every layer, each
   keeping its own norms, convolution and projections.

All four are causal, so no bidirectional wrapper is required anywhere.

## 3. Hybrid arrangement and hypothesis

The arrangement is strict alternation, one Mamba block then one DeltaNet block, repeated. The
reasoning is that the two mechanisms contribute different things and neither should be starved of
depth.

DeltaNet's state is an associative table: a fast-weight matrix whose delta rule performs an
error-correcting write, displacing the association currently stored at a key before writing the new
one. Plain additive linear attention cannot do this - colliding keys accumulate instead of being
overwritten. A cipher is, at least in part, a mapping between symbols, and any such mapping is
naturally represented as an associative memory that is written to and later read.

Mamba contributes the complementary half. `Delta` is an input-dependent gate on the state update, so
the model can decide per position how much history to carry, and the depthwise convolution gives a
local 4-gram view before the scan. That matters for alignment and segmentation: the ciphertext is
longer than the plaintext, so deciding which source symbol corresponds to the current target symbol
is a local, position-sensitive problem.

Interleaving rather than staging is chosen because the depth at which either mechanism becomes useful
is not known in advance. A staged design commits to "perception early, memory late" and gives each
mechanism only one depth range; if the useful interaction is the reverse, the model cannot express
it. Alternation makes both a local-context operator and an associative memory available at every
scale, which is the same argument that motivates interleaved hybrids in Samba and Jamba. The cost is
that neither mechanism gets a deep run of consecutive layers.

### 3.1 Falsifiable predictions

- **P1.** DeltaNet reaches higher character accuracy than Mamba at equal parameter count, because the
  delta rule matches a symbol-to-symbol associative map and the selective scan does not.
- **P2.** The hybrid is at least as good as the better of its two parents on every reported metric.
- **P3.** The hybrid's margin over both parents is larger if the cipher's mapping is document
  dependent than if it is a fixed mapping, since document-dependent mappings need both local context
  and memory.
- **P4.** The Transformer remains the strongest model at equal parameter count, and the gap to the
  linear models grows with sequence length, since the cipher task needs no long-range attention but
  the Transformer pays for it regardless.

These predictions are the ones the final report is measured against. P1 and P3 are stated strongly
enough to be wrong.

## 4. Evaluation protocol

Character-level accuracy, word-level accuracy, precision, recall, F1, Levenshtein distance, parameter
count and FLOPs, as required. Precision, recall and F1 are computed at the character level over
non-padding target positions, treating each character as a class and macro-averaging, since a
per-character mapping has no natural positive class. Word-level accuracy is exact-match over
whitespace tokens after decoding. Levenshtein distance is the mean per-sequence edit distance.

Comparisons use a matched parameter budget, the same vocabulary, the same sequence-length caps, the
same optimiser, the same step budget and the same seed set. The checkpoint with the best validation
loss is the one evaluated; character accuracy, word accuracy, precision, recall, F1 and Levenshtein
distance are computed once, after training, on that checkpoint. Validation loss is measured every 500
steps. All runs are logged to Weights and Biases.

**Milestone 0: learnability probe.** Before the four models are built, the smallest Transformer is
trained on a subsample to check that the mapping is learnable at all. The probe passes if validation
character accuracy exceeds the marginal baseline (predicting the most frequent plaintext character)
by a clear margin. If it does not, the architecture work is paused and the data question is raised,
because no mixer choice can compensate for a target that carries no information about the source.

## 5. Constraints, risks and observations

- **Memory.** All models run inference within 10 GB of VRAM. DeltaNet's state is wider than Mamba's
  and the concatenated sequence is longer than either alone, so the hybrid is the binding case.
- **Numerical stability.** `A_bar` underflows in half precision; the scan and the triangular solve
  run in fp32. `Delta` is clamped to a positive range before discretisation.
- **Padding and state reset.** Padded positions write state in both recurrences. State is reset at
  sequence boundaries, and padding is masked out of the loss and the scan.
- **Data observation.** The released training split shows no simple structural dependence between
  aligned plaintext and ciphertext characters: the mapping is not a substitution, not
  position-dependent, and not additive or autokey; identical plaintext strings receive different
  ciphertext. The length ratio is stable at about 1.131 across splits. The working assumption is a
  document-conditioned cipher, and Milestone 0 exists to test it cheaply.
