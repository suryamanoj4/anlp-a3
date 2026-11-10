# Choices

## Tokenisation

Character level, not BPE. The corpus uses 81 ciphertext symbols and 27 plaintext symbols, so the
vocabulary is 111 tokens; a subword vocabulary would be larger than the alphabet and would add
nothing. Ciphertext symbols are close to uniformly distributed (bigram entropy 12.54 of a possible
12.68 bits) and share almost no mutual information with the plaintext at any offset, so merges
learned from it would encode noise. Character level also keeps the reported metrics exact: character
accuracy, Levenshtein distance and word accuracy are all defined on characters, and the held-out
output is one string per id.

## Task formulation

A causal decoder over `[ciphertext][SEP][plaintext][EOS]`, with the loss applied only to the
plaintext segment. Ciphertext is about 1.13 times longer than plaintext, so per-position transduction
is not possible. Keeping the source in the causal prefix conditions every target token on the whole
ciphertext without cross-attention, which matters because cross-attention inside Mamba or DeltaNet
would violate the requirement that those architectures stay faithful to their papers.

## Architecture budget

`d_model=384`, `n_heads=6`, `n_layers=6`, `d_ff=1536`, giving about 10.7M parameters for the
Transformer and DeltaNet. Mamba uses `expand=4` so that a block costs the same `12 d^2` as the other
two; at the paper's default `expand=2` a Mamba stack has half the parameters at equal width and depth,
which would make the comparison a comparison of capacity rather than of architecture. The hybrid
alternates one Mamba and one DeltaNet layer, four of each.

## Mamba block composition

Mamba layers carry no separate feed-forward network. The Mamba block's projection to `2 * d_inner`
plus the gated branch already performs the expansion that an FFN provides, and the paper's block does
not include one. Transformer and DeltaNet layers do include a feed-forward network, because the
DeltaNet paper specifies a LLaMA-style block. This asymmetry is deliberate and is why the matched
parameter budget uses `expand=4` rather than an added FFN.

## Sequence and batch handling

`max_source_len=128` and `max_target_len=112`. The length distribution is highly skewed: median
document is 31 characters, mean 865, maximum 34138. Batches are formed by sorting on length, so a
long document does not pad an entire batch to its length; without this, roughly four fifths of the
processed tokens would be padding.

## Data budget

Training uses a token budget rather than the whole split. `max_tokens=45000000` covers roughly 50k
documents, which at 6000 steps of batch 32 is about 270M tokens and about 25 minutes per model on a
T4. The full split is 490M plaintext characters per epoch and would take about 1.6 hours per model,
which is not affordable. The held-out and test splits are never subsampled.

## Numerics

Training runs in fp16 autocast with a gradient scaler on Turing hardware, which has no bf16 support.
The Mamba scan and the DeltaNet triangular solve always run in fp32, because the decay product
underflows in half precision and an inverted triangular matrix in half precision is not reliable.
Mamba's scan is gradient-checkpointed, since a naive implementation retains `batch x length x
d_inner x d_state` tensors per layer and would exceed the memory of an 11 GB card.

## Checkpoint selection and evaluation timing

Training is budgeted by step count, not wall clock, so the number of optimisation steps is identical on
any hardware and the comparison is not affected by machine speed. Wall-clock time is recorded and
reported as an outcome.

Checkpoints are selected by validation loss, measured every 500 steps. Validation loss is a
teacher-forced pass over a fixed set of validation batches: it costs seconds, has low variance, and is
available at every checkpoint. Character accuracy, word accuracy, precision, recall, F1 and Levenshtein
distance require autoregressive generation, which is roughly an order of magnitude more expensive and
much noisier at fifty samples, so those metrics are computed once after training on the selected
checkpoint. The training loop therefore logs loss, learning rate, throughput and validation loss to
Weights and Biases, and the metrics table is produced by `src/part1/evaluate.py`.

## Metrics

Character accuracy, word accuracy, macro precision/recall/F1 over characters, mean Levenshtein
distance, parameter count and analytic FLOPs. Macro averaging is used because a symbol-to-symbol
mapping has no natural positive class. FLOPs are analytic and reported for a single forward pass at
the maximum sequence length.

## Measured throughput on the target hardware

Measured on a Colab T4 with fp16 autocast, batch 16, length-bucketed batches. These numbers set the
step budget, and they replace the earlier estimates.

| Architecture | Tokens per second | Approximate time for 3000 steps |
|---|---|---|
| Transformer | ~64000 | ~2 min |
| DeltaNet | ~4000 | ~25 min |
| Mamba, faithful settings | ~1900 | ~57 min |
| Mamba, tuned settings | ~4100 | ~26 min |
| Hybrid | ~1900 | ~50 min |

The Transformer is an order of magnitude faster per token than the recurrent mixers at these
sequence lengths. Sequences are short (about 134 tokens after bucketing), so the quadratic attention
term is negligible and the linear-time architectures pay their sequential-scan cost without
collecting their long-context benefit. This is a property of the task and hardware, not of the
implementations, and it is reported rather than hidden.

## Mamba deviations forced by memory and time

Two deviations from the paper's defaults are used, both recorded here:

- `d_state=8` instead of 16. This changes only the size of the `A` parameter and the state width; it
  is the difference between fitting in 14.5 GB and not fitting.
- The scan is not gradient-checkpointed, and `scan_chunk=32` instead of 16. Checkpointing a naive
  chunked scan roughly doubles the time and was measured at 1883 tokens per second; without it, and
  with 32-step chunks, the same model reaches about 4147 tokens per second. The larger chunk is safe
  because with `d_state=8` the accumulated decay stays within fp32 range.

`max_steps=3000` with `batch_size=16` is therefore the budget that all four architectures can
complete in roughly the same wall-clock window. Steps and batch size are held equal across
architectures so that the comparison is of architectures rather than of optimisation budgets; tokens
seen and FLOPs are reported alongside.

