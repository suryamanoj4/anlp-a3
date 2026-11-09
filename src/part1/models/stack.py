import torch.nn as nn


class MixerLayer(nn.Module):
    def __init__(self, mixer, d_model, d_ff, dropout, use_ffn=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mixer = mixer
        self.use_ffn = use_ffn
        if use_ffn:
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

    def forward(self, x, attention_mask=None):
        x = x + self.mixer(self.ln1(x))
        if self.use_ffn:
            x = x + self.mlp(self.ln2(x))
        return x


class MixerLM(nn.Module):
    def __init__(self, vocab_size, d_model, mixers, d_ff=None, dropout=0.0):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                MixerLayer(mixer, d_model, d_ff, dropout, use_ffn=use_ffn)
                for mixer, use_ffn in mixers
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask):
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.lm_head(self.ln_f(x))
