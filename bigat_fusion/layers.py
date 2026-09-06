"""Neural network layers used by BiGAT-Fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMoEDecoder(nn.Module):
    """Combine nonlinear and low-rank bilinear association scores."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.2, rank: int = None):
        super().__init__()
        rank = max(8, dim // 2) if rank is None else rank
        self.mlp = nn.Sequential(
            nn.Linear(4 * dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.U = nn.Linear(dim, rank, bias=False)
        self.V = nn.Linear(dim, rank, bias=False)
        self.w = nn.Parameter(torch.randn(rank))
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.normal_(self.w, mean=0.0, std=0.02)
        self._t = nn.Parameter(torch.tensor(0.0))
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, drug_h, disease_h, drug_idx, disease_idx, drug_bias, disease_bias):
        features = torch.cat(
            [drug_h, disease_h, drug_h * disease_h, (drug_h - disease_h).abs()], dim=-1
        )
        mlp_score = self.mlp(features).squeeze(-1)
        bilinear_score = (self.U(drug_h) * self.V(disease_h)) @ self.w
        bilinear_score = (F.softplus(self._t) + 1e-3) * bilinear_score
        gate_input = torch.cat(
            [drug_h * disease_h, (drug_h - disease_h).abs()], dim=-1
        )
        mixture = 0.5 * torch.sigmoid(self.gate(gate_input)).squeeze(-1)
        return (
            mlp_score
            + mixture * bilinear_score
            + drug_bias(drug_idx).squeeze(-1)
            + disease_bias(disease_idx).squeeze(-1)
            + self.b0
        )


class GATLayer(nn.Module):
    """Graph attention layer for a homogeneous similarity graph."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, alpha: float = 0.2):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        self.a = nn.Parameter(torch.empty(2 * out_dim, 1))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, features, edge_src, edge_dst):
        node_count = features.size(0)
        hidden = features @ self.W
        hidden = F.dropout(hidden, p=self.dropout, training=True) if self.dropout and self.training else hidden
        source_h, target_h = hidden[edge_src], hidden[edge_dst]
        energy = self.leaky_relu(
            torch.matmul(torch.cat([target_h, source_h], dim=1), self.a)
        ).squeeze(1)
        energy = energy - energy.max()
        exp_energy = torch.exp(energy)
        denominator = torch.zeros(node_count, device=features.device).index_add_(
            0, edge_dst, exp_energy
        )
        attention = exp_energy / (denominator[edge_dst] + 1e-16)
        if self.dropout:
            attention = F.dropout(attention, p=self.dropout, training=self.training)
        output = torch.zeros_like(hidden).index_add_(
            0, edge_dst, attention.unsqueeze(1) * source_h
        )
        return F.relu(output)


class BiGATLayer(nn.Module):
    """Bidirectional graph attention layer for the association graph."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, alpha: float = 0.2):
        super().__init__()
        self.W_drug = nn.Parameter(torch.empty(in_dim, out_dim))
        self.W_dis = nn.Parameter(torch.empty(in_dim, out_dim))
        self.a_drug = nn.Parameter(torch.empty(2 * out_dim, 1))
        self.a_dis = nn.Parameter(torch.empty(2 * out_dim, 1))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = dropout
        for parameter in [self.W_drug, self.W_dis, self.a_drug, self.a_dis]:
            nn.init.xavier_uniform_(parameter)
        self.last_alpha_d = None
        self.last_alpha_p = None

    def _attention(self, source_h, target_h, target_idx, target_count, vector):
        energy = self.leaky_relu(
            torch.matmul(torch.cat([target_h, source_h], dim=1), vector)
        ).squeeze(1)
        energy = energy - energy.max()
        exp_energy = torch.exp(energy)
        denominator = torch.zeros(target_count, device=source_h.device).index_add_(
            0, target_idx, exp_energy
        )
        attention = exp_energy / (denominator[target_idx] + 1e-16)
        if self.dropout:
            attention = F.dropout(attention, p=self.dropout, training=self.training)
        return attention

    def forward(
        self,
        drug_x,
        disease_x,
        disease_to_drug_src,
        disease_to_drug_dst,
        drug_to_disease_src,
        drug_to_disease_dst,
    ):
        drug_h = drug_x @ self.W_drug
        disease_h = disease_x @ self.W_dis
        if self.dropout:
            drug_h = F.dropout(drug_h, p=self.dropout, training=self.training)
            disease_h = F.dropout(disease_h, p=self.dropout, training=self.training)

        drug_attention = self._attention(
            disease_h[disease_to_drug_src],
            drug_h[disease_to_drug_dst],
            disease_to_drug_dst,
            drug_x.size(0),
            self.a_drug,
        )
        drug_output = torch.zeros_like(drug_h).index_add_(
            0,
            disease_to_drug_dst,
            drug_attention.unsqueeze(1) * disease_h[disease_to_drug_src],
        )

        disease_attention = self._attention(
            drug_h[drug_to_disease_src],
            disease_h[drug_to_disease_dst],
            drug_to_disease_dst,
            disease_x.size(0),
            self.a_dis,
        )
        disease_output = torch.zeros_like(disease_h).index_add_(
            0,
            drug_to_disease_dst,
            disease_attention.unsqueeze(1) * drug_h[drug_to_disease_src],
        )
        self.last_alpha_d = drug_attention.detach()
        self.last_alpha_p = disease_attention.detach()
        return F.relu(drug_output), F.relu(disease_output)
