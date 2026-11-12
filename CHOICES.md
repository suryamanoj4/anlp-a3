# Choices

## Tokenisation

Character level, not BPE. After the preprocessing below the source alphabet is 63 symbols (space,
digits, upper and lower case) and the target alphabet is 27, giving a vocabulary of 94 tokens
including padding, separator, end and unknown. A subword vocabulary would be larger than the alphabet
and would add nothing. The mapping carries no usable signal (measured below), so merges learned from
it would encode noise. Character level also keeps the reported metrics exact: character accuracy,
Levenshtein distance and word accuracy are all defined on characters, and the held-out output is one
string per id.

## Punctuation removal and unknown tokens

The ciphertext contains 18 punctuation symbols (`!'+,-./:;<=>?[]_{}`) that are inserted noise. Removing
them makes ciphertext and plaintext lengths match for 100 percent of training pairs, with a mean length
ratio of exactly 1.0000 instead of 1.1311. Roughly 13 percent of source tokens were pure padding
otherwise. Removal is applied inside the tokeniser's source encoder, so it is identical during training
and during held-out inference rather than being a one-off preprocessing script.

A dedicated unknown token is used for source characters outside the vocabulary. Previously an unseen
character was silently mapped to whichever source symbol sorts first, which would corrupt held-out
input; now it is an explicit token. Unknown target tokens are dropped during decoding.

## Patch training

Removing punctuation makes the cipher-to-plaintext mapping exactly one symbol per symbol, and the
corpus shows no cross-character dependence: within-document mapping purity is 0.0679 against 0.0678 for
a shuffled control, i.e. indistinguishable. There is therefore no long-range structure to preserve, so
documents are split into aligned patches of 128 characters. A patch is `128 source` followed by
`128 target`, and the two are aligned by construction because the stripped ciphertext has the same
length as its plaintext.

This replaces truncating each document to its first 128 characters, which discarded 93 percent of the
corpus. The 12,000-document slice contains 89,078 patches, about 23 million tokens per epoch, against
1.57 million when whole documents were truncated. Every character is covered exactly once per epoch,
memory is bounded by the patch size rather than by the longest document, and padding waste disappears.

The same property makes evaluation simpler: a document is decoded by running consecutive 128-character
patches and concatenating the outputs, so long test documents can be scored in full instead of being
skipped. Patches are capped at 32 per document during evaluation to bound the cost.

The trade-off is that a patch cannot see beyond its own boundaries. That is acceptable only because the
mapping has been measured to be position-independent; if the cipher had any cross-character dependence,
this design would discard it. This assumption is reported rather than left implicit.

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
much noisier at fifty samples, so those metrics are computed once, at the end of training, on the best
checkpoint.

That final evaluation runs inside the training process, so a single Weights and Biases run contains the
training curves, the validation-loss curve and the test metrics together. A separate standalone entry
point, `src/part1/evaluate.py`, re-scores a saved checkpoint without retraining, which is useful if a
model needs to be re-evaluated on a different split or limit. The training loop logs loss, learning
rate, throughput and validation loss, then the test metrics under a `test/` prefix.

## Metrics

Character accuracy, word accuracy, macro precision/recall/F1 over characters, mean Levenshtein
distance, parameter count and analytic FLOPs. Macro averaging is used because a symbol-to-symbol
mapping has no natural positive class. FLOPs are analytic and reported for a single forward pass at
the maximum sequence length.

## Fused selective scan in Triton

The eager scan materialises the discretised recurrence as `batch x time x d_inner x d_state` tensors
and walks the sequence in Python-level chunks. At `d_inner=1536`, `d_state=8`, `batch=16` and
`time=240` that is roughly 47 million float32 elements per intermediate, and the measured efficiency is
about 0.5 percent of the device peak, because the work is memory traffic rather than arithmetic.

The Triton kernel assigns one program to each `(batch, channel)` pair. The `N`-dimensional state for
that channel is carried in registers across the whole sequence, so no intermediate state tensor is
written:

```
h = 0
for t in range(T):
    a  = exp(delta[t] * A)
    h  = a * h + delta[t] * B[t] * x[t]
    y[t] = sum(h * C[t])
```

Forward writes only the output and the state history. Backward runs the adjoint recurrence in reverse,
`g_t = C_t * dy_t + a_{t+1} * g_{t+1}`, from which the gradients follow as `dx_t = delta_t * (B_t . g_t)`,
`ddelta_t = g_t . (A * a_t * h_{t-1} + B_t * x_t)`, `dB_t = g_t * delta_t * x_t`, `dC_t = dy_t * h_t`,
and `dA = sum_t g_t * delta_t * a_t * h_{t-1}`, the last accumulated across the batch with atomics
because `A` is shared.

Everything is computed in float32 regardless of autocast, because the decay product underflows in half
precision. `src/part1/check_scan.py` compares the kernel against the eager implementation on both the
forward output and every parameter gradient before the flag is enabled.

## Measured throughput on the target hardware

Measured on a Colab T4 with fp16 autocast, batch 16, length-bucketed batches. These numbers set the
step budget, and they replace the earlier estimates.

| Architecture | Scan implementation | Tokens per second | 9200 steps |
|---|---|---|---|
| Transformer | eager | 42131 | 7.7 min |
| DeltaNet | eager | 16657 | 19.6 min |
| Mamba | eager | 3862 | 84.3 min |
| Mamba | **triton** | **8384** | **38.8 min** |
| Hybrid | eager | 6682 | 48.7 min |
| Hybrid | **triton** | **11306** | **28.8 min** |

The fused scan gives 2.17 times on Mamba and 1.69 times on the hybrid. Training loss after 100 steps is
identical to four decimal places between the eager and fused paths (2.4908 and 2.4536 respectively),
which is the in-situ confirmation of the gradient parity test. With the fused scan the four-model sweep
costs about 1 hour 46 minutes instead of 2 hours 57 minutes.

`torch.compile` was measured and rejected: it raised a Dynamo error on the Transformer and spent more
than five minutes compiling DeltaNet at zero GPU utilisation, which is not acceptable for a one-shot
training run. The flag remains in the code but is disabled in every configuration.

The Transformer remains roughly five times faster per token than the fastest recurrent mixer.
Sequences are short (about 134 tokens after bucketing), so the quadratic attention term is negligible
and the linear-time architectures pay their sequential-scan cost without collecting their long-context
benefit. This is a property of the task and hardware, not of the implementations, and it is reported
rather than hidden.

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


# Part 2 choices

## Which Bi-GCN

"Bi-GCN" in this assignment refers to Bian et al., *Rumor Detection on Social Media with Bi-Directional
Graph Convolutional Networks* (AAAI 2020, arXiv 2001.06362). A second, unrelated paper also calls
itself BiGCN: Chen et al., *BiGCN: A Bi-directional Low-Pass Filtering Graph Neural Network*
(arXiv 2101.05519). That model filters a node graph and a learned latent feature graph in two
directions; it is a node-classification method for a single attributed graph and has no notion of a
root post or a propagation tree. Since this assignment is about propagation graphs, Bian et al. is the
matching reference, and it is also the BiGCN baseline used in the UPFD literature.

The implementation follows Bian et al. directly:

- top-down GCN on the parent-to-child adjacency and bottom-up GCN on its transpose, with separate
  weights and two layers each;
- root feature enhancement, concatenating the root hidden vector of the previous layer to every node
  at every layer;
- DropEdge on the directed edges at rate 0.2, then the transpose for the bottom-up branch;
- mean pooling over each direction, concatenation of the two pooled vectors, and a linear classifier.

## Dataset and split

PyG's `UPFD` loader with `name="gossipcop"` downloads and caches under `data/part2`. The provided
20/10/70 split is used exactly as shipped: `train.pt`, `val.pt`, `test.pt`. No resampling, no
re-splitting, and no merging of splits.

## Edge orientation

UPFD stores edges with a fixed direction per dataset but does not document which. The loader checks,
over the first 256 training graphs, whether edges point into the local root node 0. If they do, the
stored direction is child-to-parent and the edge index is flipped once per batch so that the top-down
branch always aggregates parent-to-child. The inferred direction is printed at the start of every run
and recorded in `results/part2/details.json`. The same check also reports the fraction of edges whose
reverse is present, so an accidentally undirected graph would be visible in the log rather than
silently collapsing the two branches.

## Feature standardisation

Each feature type is standardised per dimension with the mean and standard deviation of the training
split only. The transform is identical for all four feature types and for all architectures; only the
input data changes. This keeps the ten-dimensional profile features and the 768-dimensional BERT
features on comparable scales without giving any configuration a different preprocessing step. A
dimension whose training standard deviation is below `1e-6` is left unscaled.

## Metrics

The task is binary. Accuracy is reported as-is, and precision, recall and F1 are **macro-averaged over
the two classes**, so the numbers do not depend on which label is called positive. Per-class counts and
the binary (positive = class 1) metrics are also written to `results/part2/details.json` for the report,
but the values in `results/part2/results.json` and `output/part2/results.json` are the macro means.

Five seeds are used, `0` to `4`. The reported deviation is the population standard deviation across
the five test scores (`ddof=0`), not the standard error.

## Training protocol

One protocol is shared by every architecture and every feature type so that the comparison is of models
and not of tuning budgets:

| Setting | Value |
|---|---|
| Hidden width | 128 |
| Layers | 2 |
| Dropout | 0.5 |
| DropEdge (Bi-GCN only) | 0.2 |
| GAT heads | 4 on the first layer, 1 on the second |
| Optimiser | Adam, lr 1e-3, weight decay 5e-4 |
| Batch size | 32 |
| Maximum epochs | 100 |
| Early stopping | validation macro-F1, patience 20 |
| Checkpoint | the epoch with the best validation macro-F1 |

The best-validation-F1 checkpoint is the one evaluated on test, as the assignment requires. It is not
the best-loss checkpoint. Training is seeded with `random`, `numpy` and `torch` together, cuDNN runs in
deterministic mode, and the shuffling generator is seeded per run.

The Bi-GCN root enhancement doubles the width entering each layer and the pooled representation, which
is a property of the published model rather than a tuning choice; its parameter count is reported
alongside the others so the comparison is not hidden.

## Deduplicated configurations

`GraphSAGE` with BERT features is both the `sage` row of the architecture comparison and the `bert`
row of the feature ablation. It is trained once and used for both rows, because the protocol, the seed
and the data are identical. Writing the same number twice from two independent runs would be
redundant.

## Entry point

`bash ./scripts/part2/test_gnn.sh` runs the full grid: five architectures on BERT, plus GraphSAGE on
profile, spaCy, BERT and content, five seeds each, and writes `output/part2/results.json`.

`bash ./scripts/part2/test_gnn.sh --ta` runs the same pipeline with the first seed only and writes
`output/part2/results_ta.json`. The `--ta` variant exists so the marker can verify the full pipeline
end-to-end in minutes rather than in the time the full grid takes; it is not the submitted result.

## Experiment tracking

WandB project `anlp-a3-part2`, one run per configuration named `part2-<arch>-<feature>` for training
curves, and a matching `part2-<arch>-<feature>-evals` run for the test metrics and their mean and
standard deviation across seeds.
