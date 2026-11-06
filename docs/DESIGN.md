# Design: Learning the Dominion Cipher

Status: written and committed before any Part A implementation. Not modified afterwards.

## 1. Task framing

The data is a set of aligned plaintext/ciphertext character sequences. The goal is to learn the
mapping and apply it to unseen ciphertext, not to model plaintext likelihood. Three framings were
considered:

- **Causal decoder over ciphertext, predicting the next plaintext character.** Rejected: it
  conditions on the plaintext prefix while predicting the same position, so training-time accuracy
  is dominated by copying rather than by the cipher. Reported decryption quality would not transfer
  to held-out sequences.
- **Encoder-decoder with cross-attention.** Rejected: cross-attention inside a Mamba or DeltaNet
  decoder is attention interleaved into the recurrence, which the assignment forbids, and it would
  make the four models non-comparable.
- **Bidirectional per-position transducer.** Adopted. A bidirectional stack encodes the ciphertext,
  an
d a per-position head classifies each output character over the character vocabulary. Every
  position is predicted from the full ciphertext context and never from the target prefix.

All four models therefore share one interface: `(B, L) -> (B, L, V)` token ids in, logits out,
with padding masked 
out of the loss. Mamba and DeltaNet are causal by construction, so each is
wrapped bidirectionally: run the recurrence forward and backward, concatenate the two states, and
project back to the model width. The wrapper and the head are identical across models; only the
mixer differs. This is what makes the comparison meaningful.

Positional information is supplied to the Transformer with learned absolute embeddings. The
recurrences encode order intrinsically, so no extra embedding is added. This asymmetry is a known
confound and is addressed in the analysis rather than hidden.

## 2. Models

1. **Transformer** - a bidirectional encoder with hand-written multi-head self-attention
   (scaled dot-product, causal mask disabled, padding mask enabled), pre-norm residual blocks and a
   GELU FFN. Quadratic in sequence length; the accuracy ceiling the linear models are judged against.
2. **Mamba** - selective SSM blocks: real diagonal `A` initialised as `-(1..N)`, input-dependent
   `B`, `C` and `Delta`, ZOH discretisation as `A_bar = exp(Delta * A)`,
   `B_bar = (exp(Delta*A) - 1)/A * B`, depthwise causal convolution with kernel 4, SiLU gating with a
   parallel branch `z`, associative scan in log space for the recurrence, fp32 accumulation.
3. **DeltaNet** - chunkwise delta-rule linear attention with chunk size 64. State is a per-head
   fast-weight matrix; the transition is `S <- S(I - beta k k^T) + beta v k^T` with
   `beta = sigmoid(W_beta x)`, implemented through the chunk-level inverse of
   `I + tril(diag(beta) K K^T, -1)` by forward substitution, with L2-normalised `SiLU`-activated
   q and k and RMSNorm before the output projection. Vanilla delta rule, no decay gate.
4. **Mamba-DeltaNet hybrid** - the two mixers interleaved 1:1, alternating at every layer, with each
   mixer keeping its own norms, convolution and projections.

## 3. Hybrid arrangement and hypothesis

The arrangement is strict alternation, one Mamba block then one DeltaNet block, repeated. The
reasoning is that the two mechanisms contribute different things and neither should be starved of
depth.

A substitution cipher is an associative table: a key character maps to a value character. DeltaNet's
state is exactly such a table - a fast-weight matrix whose delta rule performs an error-correcting
write, displacing the association currently stored at a key before writing the new one. Plain
additive linear attention cannot do this: colliding keys accumulate instead of being overwritten. For
a task whose ground truth is a lookup with a bounded number of keys, that overwrite semantics is the
right inductive bias.

Mamba contributes the complementary half. `Delta` is an input-dependent gate on the state update, so
the model can decide per position how much history to carry, and the depthwise convolution gives a
local 4-gram view before the scan. That matters if the protocol is not a pure substitution but a
context-dependent one - polyalphabetic, homophonic, or position-dependent - where the mapping depends
on the surrounding characters rather than on the character alone.

Interleaving rather than staging is chosen because the depth at which a cipher's context dependence
becomes visible is not known in advance. A staged design commits to "perception early, memory late"
and gives each mechanism only one depth range; if the useful interaction is the reverse, the model
cannot express it. Alternation makes both a local-context operator and an associative memory
available at every scale, which is the same argument that motivates interleaved hybrids in Samba and
Jamba. The cost is that neither mechanism gets a deep run of consecutive layers.

### 3.1 Falsifiable predictions

- **P1.** On a pure substitution cipher, DeltaNet reaches higher character accuracy than Mamba at
  equal parameter count, because the delta rule matches a lookup table and the selective scan does not.
- **P2.** The hybrid is at least as good as the better of its two parents on every reported metric.
- **P3.** The hybrid's margin over both parents is larger if the cipher is context-dependent than if
  it is a pure substitution. If the margin is uniform, interleaving is doing something other than
  combining context and memory.
- **P4.** The Transformer remains the strongest model at equal parameter count, and the gap between
  the Transformer and the linear models grows with sequence length, since the cipher task needs no
  long-range attention but the Transformer pays for it regardless.

These predictions are the ones the final report is measured against. P1 and P3 in particular are
stated strongly enough to be wrong.

## 4. Evaluation protocol

Character-level accuracy, word-level accuracy, precision, recall, F1, Levenshtein distance, parameter
count and FLOPs, as required. Precision, recall and F1 are computed at the character level over the
non-padding positions, treating each character as a class and macro-averaging, since a per-character
substitution has no natural positive class. Word-level accuracy is exact-match over whitespace
tokens after decoding. Levenshtein distance is reported as the mean per-sequence edit distance.

Comparisons are run at a matched parameter budget within 10 percent, with the same vocabulary,
sequence length cap, optimiser, schedule and seed set. The checkpoint with the best validation
character accuracy is the one evaluated, matching the protocol required in Part B. All runs are
logged to Weights and Biases.

## 5. Constraints and risks

- **Memory.** All models must run inference within 10 GB of VRAM. The bidirectional wrapper doubles
  the state and scan cost per layer, and DeltaNet's state is wider than Mamba's, so the hybrid is the
  binding case. Width and depth are chosen against this budget, not against the Transformer.
- **Numerical stability.** `A_bar` underflows in bf16; the scan and the chunk-level inverse run in
  fp32. `Delta` is clamped to a positive range before discretisation.
- **Padding leakage.** Padded positions write state in both recurrences. State is reset at sequence
  boundaries and padded `k`, `v` and `beta` are zeroed, rather than only masking the output.
- **Frozen document.** This file is committed before implementation begins and is not edited after.
  If the observed data contradicts an assumption here - in particular that ciphertext and plaintext
  are position-aligned and equal in length - that is reported as a finding rather than fixed by
  rewriting the hypothesis.
