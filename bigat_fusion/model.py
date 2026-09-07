"""BiGAT-Fusion model assembly."""

import torch
import torch.nn as nn

from .layers import BiGATLayer, GATLayer, ResidualMoEDecoder


def edge_index(neighbor_lists):
    """Convert adjacency lists to source and destination tensors."""
    source, destination = [], []
    for target, neighbors in enumerate(neighbor_lists):
        for neighbor in neighbors:
            source.append(neighbor)
            destination.append(target)
    return torch.tensor(source, dtype=torch.long), torch.tensor(destination, dtype=torch.long)


def bipartite_edge_index(drug_neighbors, disease_neighbors):
    """Create directed edge tensors for both association graph directions."""
    disease_to_drug_src, disease_to_drug_dst = [], []
    for drug, diseases in drug_neighbors.items():
        for disease in diseases:
            disease_to_drug_src.append(disease)
            disease_to_drug_dst.append(drug)

    drug_to_disease_src, drug_to_disease_dst = [], []
    for disease, drugs in disease_neighbors.items():
        for drug in drugs:
            drug_to_disease_src.append(drug)
            drug_to_disease_dst.append(disease)

    return {
        "dis_to_drug_src": torch.tensor(disease_to_drug_src, dtype=torch.long),
        "dis_to_drug_dst": torch.tensor(disease_to_drug_dst, dtype=torch.long),
        "drug_to_dis_src": torch.tensor(drug_to_disease_src, dtype=torch.long),
        "drug_to_dis_dst": torch.tensor(drug_to_disease_dst, dtype=torch.long),
    }


class BiGATFusionModel(nn.Module):
    """Fuse similarity-view and association-topology representations."""

    def __init__(
        self,
        n_drugs: int,
        n_diseases: int,
        drug_feat_neighbors: list,
        disease_feat_neighbors: list,
        drug_neighbors: dict,
        disease_neighbors: dict,
        *,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.drug_emb = nn.Embedding(n_drugs, embed_dim)
        self.disease_emb = nn.Embedding(n_diseases, embed_dim)
        self.gat_drug_feat = GATLayer(embed_dim, embed_dim, dropout)
        self.gat_dis_feat = GATLayer(embed_dim, embed_dim, dropout)
        self.bipartite_gat = BiGATLayer(embed_dim, embed_dim, dropout)
        self.gate_drug = nn.Linear(embed_dim * 2, 1)
        self.gate_dis = nn.Linear(embed_dim * 2, 1)
        nn.init.constant_(self.gate_drug.bias, 0.0)
        nn.init.constant_(self.gate_dis.bias, 0.0)

        self.bias_d = nn.Embedding(n_drugs, 1)
        self.bias_p = nn.Embedding(n_diseases, 1)
        nn.init.zeros_(self.bias_d.weight)
        nn.init.zeros_(self.bias_p.weight)
        self.decoder = ResidualMoEDecoder(embed_dim, hidden_dim, dropout)
        self._register_edge_buffers(
            drug_feat_neighbors,
            disease_feat_neighbors,
            drug_neighbors,
            disease_neighbors,
        )

    def _register_edge_buffers(
        self,
        drug_feat_neighbors,
        disease_feat_neighbors,
        drug_neighbors,
        disease_neighbors,
    ):
        drug_feat_src, drug_feat_dst = edge_index(drug_feat_neighbors)
        disease_feat_src, disease_feat_dst = edge_index(disease_feat_neighbors)
        mapping = {
            "drug_feat_src": drug_feat_src,
            "drug_feat_dst": drug_feat_dst,
            "disease_feat_src": disease_feat_src,
            "disease_feat_dst": disease_feat_dst,
            **bipartite_edge_index(drug_neighbors, disease_neighbors),
        }
        for name, tensor in mapping.items():
            self.register_buffer(name, tensor)

    def forward(self):
        drug_initial = self.drug_emb.weight
        disease_initial = self.disease_emb.weight
        drug_feature = self.gat_drug_feat(
            drug_initial, self.drug_feat_src, self.drug_feat_dst
        )
        disease_feature = self.gat_dis_feat(
            disease_initial, self.disease_feat_src, self.disease_feat_dst
        )
        drug_topology, disease_topology = self.bipartite_gat(
            drug_initial,
            disease_initial,
            self.dis_to_drug_src,
            self.dis_to_drug_dst,
            self.drug_to_dis_src,
            self.drug_to_dis_dst,
        )

        drug_gate = torch.sigmoid(
            self.gate_drug(torch.cat([drug_feature, drug_topology], dim=1))
        )
        disease_gate = torch.sigmoid(
            self.gate_dis(torch.cat([disease_feature, disease_topology], dim=1))
        )
        drug_fused = drug_gate * drug_feature + (1 - drug_gate) * drug_topology
        disease_fused = (
            disease_gate * disease_feature + (1 - disease_gate) * disease_topology
        )
        return drug_fused, disease_fused

    def logits_on_pairs(self, drug_idx, disease_idx):
        drug_z, disease_z = self.forward()
        return self.decoder(
            drug_z[drug_idx],
            disease_z[disease_idx],
            drug_idx,
            disease_idx,
            self.bias_d,
            self.bias_p,
        )

    def get_fused_embeddings(self):
        return self.forward()
