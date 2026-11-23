"""Path encoder, mass-aware patient pooling, and two edge-prediction heads."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def positional_encoding(length, width, device, dtype):
    positions = torch.arange(length, device=device, dtype=dtype)[:, None]
    frequency = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype)
                          * (-math.log(10000.0) / width))
    encoding = torch.zeros(length, width, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(positions * frequency)
    encoding[:, 1::2] = torch.cos(positions * frequency)
    return encoding


class TrajectoryTransformer(nn.Module):
    def __init__(self, n_states, config):
        super().__init__()
        self.config = dict(config)
        self.n_states = n_states
        width = config["d_model"]
        heads = config["n_heads"]
        if width % 2 or width % heads:
            raise ValueError("d_model must be even and divisible by n_heads")
        self.width = width
        half = width // 2
        self.state_embedding = nn.Embedding(n_states, half)
        self.transition_embedding = nn.Sequential(
            nn.Linear(3, half), nn.GELU(), nn.Linear(half, half))
        layer = nn.TransformerEncoderLayer(
            width, heads, config["ffn_dim"], config["dropout"],
            activation="gelu", batch_first=True, norm_first=True)
        self.path_encoder = nn.TransformerEncoder(
            layer, config["encoder_layers"], enable_nested_tensor=False)
        self.summary_queries = nn.Parameter(torch.randn(config["summary_slots"], width) * 0.02)
        self.pool_q = nn.Linear(width, width)
        self.pool_k = nn.Linear(width, width)
        self.pool_v = nn.Linear(width, width)
        self.edge_embedding = nn.Linear(width, width)
        self.gap_embedding = nn.Sequential(
            nn.Linear(1, half), nn.GELU(), nn.Linear(half, width))
        self.edge_norm = nn.LayerNorm(width)
        self.summary_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
            width, 1, dropout=config["dropout"], batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, config["ffn_dim"]), nn.GELU(),
            nn.Linear(config["ffn_dim"], width))
        self.dropout = nn.Dropout(config["dropout"])
        self.ot_head = nn.Linear(width, 1)
        self.structure_head = nn.Linear(width, 1)
        # Begin near gates = 0.5 rather than a saturated sparse graph.
        nn.init.zeros_(self.structure_head.weight)
        nn.init.zeros_(self.structure_head.bias)

    def encode_paths(self, sample):
        states, transition = sample["states"], sample["transition"]
        outputs = []
        chunk_size = self.config["path_chunk_size"]
        for start in range(0, len(states), chunk_size):
            tokens = torch.cat([
                self.state_embedding(states[start:start + chunk_size]),
                self.transition_embedding(transition[start:start + chunk_size]),
            ], dim=-1)
            tokens = tokens + positional_encoding(
                tokens.shape[1], self.width, tokens.device, tokens.dtype)
            outputs.append(self.path_encoder(tokens).mean(dim=1))
        return torch.cat(outputs)

    def edge_queries(self, days):
        k, steps = self.n_states, len(days) - 1
        ids = torch.arange(k, device=days.device)
        source = self.state_embedding(ids).repeat_interleave(k, dim=0)
        target = self.state_embedding(ids).repeat(k, 1)
        pair = self.edge_embedding(torch.cat([source, target], dim=-1))
        position = positional_encoding(steps, self.width, days.device, pair.dtype)
        gap = self.gap_embedding(
            ((days[1:] - days[:-1]) / self.config["time_scale"])[:, None])
        return (pair[None] + position[:, None] + gap[:, None]).reshape(-1, self.width)

    def features(self, sample):
        paths = self.encode_paths(sample)
        scores = self.pool_q(self.summary_queries) @ self.pool_k(paths).T / math.sqrt(self.width)
        attention = torch.softmax(scores + sample["weights"].log()[None, :], dim=-1)
        summary = attention @ self.pool_v(paths)
        edges = self.edge_queries(sample["days"])
        normalized_summary = self.summary_norm(summary)[None]
        update, _ = self.cross_attention(
            self.edge_norm(edges)[None], normalized_summary, normalized_summary,
            need_weights=False)
        hidden = edges + self.dropout(update[0])
        return hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))

    def predictions(self, hidden, sample):
        shape = sample["target"].shape
        logits = self.ot_head(hidden).reshape(shape)
        log_prob = F.log_softmax(logits.flatten(1), dim=1).reshape(shape)
        return {
            "log_prob": log_prob,
            "prob": log_prob.exp(),
            "gates": self.structure_head(hidden).sigmoid().flatten(),
        }

    def forward(self, sample):
        return self.predictions(self.features(sample), sample)


def losses(prediction, sample, complexity_weight):
    target = sample["target"]
    kl = F.kl_div(prediction["log_prob"], target, reduction="sum") / len(target)
    gates = prediction["gates"]
    coverage = (sample["weights"] * gates[sample["edge_indices"]].prod(dim=1)).sum()
    positive_edges = target.flatten() > 0
    complexity = gates[positive_edges].mean()
    structure = 1 - coverage + complexity_weight * complexity
    return {"ot": kl, "structure": structure, "soft_coverage": coverage,
            "complexity": complexity}
