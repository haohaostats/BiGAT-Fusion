
#!/usr/bin/env python3

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Residual-MoE decoder ----
class ResidualMoEDecoder(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.2, rank: int = None):
        super().__init__()
        if rank is None:
            rank = max(8, dim // 2)

        self.mlp = nn.Sequential(
            nn.Linear(4 * dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.U = nn.Linear(dim, rank, bias=False)
        self.V = nn.Linear(dim, rank, bias=False)
        self.w = nn.Parameter(torch.randn(rank))
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.normal_(self.w, mean=0.0, std=0.02)

        self._t = nn.Parameter(torch.tensor(0.0))  # tau = softplus(t) + 1e-3

        self.gate = nn.Sequential(
            nn.Linear(2 * dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, hd, hp, d_idx, p_idx, bias_d, bias_p):
    
        z = torch.cat([hd, hp, hd * hp, (hd - hp).abs()], dim=-1)  # [B, 4D]
        s_mlp = self.mlp(z).squeeze(-1)                            # [B]

        u = self.U(hd)                                             # [B, r]
        v = self.V(hp)                                             # [B, r]
        s_bilin = (u * v) @ self.w                                 # [B]
        temp = F.softplus(self._t) + 1e-3
        s_bilin = temp * s_bilin

        g_in = torch.cat([hd * hp, (hd - hp).abs()], dim=-1)       # [B, 2D]
        gamma = 0.5 * torch.sigmoid(self.gate(g_in)).squeeze(-1)   # [B]

        logits = s_mlp + gamma * s_bilin \
               + bias_d(d_idx).squeeze(-1) + bias_p(p_idx).squeeze(-1) + self.b0
        return logits

# ---- GAT for feature view (per-domain) ----

class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, alpha: float = 0.2):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.a = nn.Parameter(torch.Tensor(2 * out_dim, 1))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, edge_src, edge_dst):
        """x: [N, D]; edges are given as src->dst indices (per-destination softmax)."""
        N = x.size(0)
        h = x @ self.W
        if self.dropout:
            h = F.dropout(h, p=self.dropout, training=self.training)

        h_src, h_dst = h[edge_src], h[edge_dst]
        e = self.leaky_relu(torch.matmul(torch.cat([h_dst, h_src], dim=1), self.a)).squeeze(1)

        e = e - e.max()
        exp_e = torch.exp(e)
        denom = torch.zeros(N, device=x.device).index_add_(0, edge_dst, exp_e)
        alpha = exp_e / (denom[edge_dst] + 1e-16)
        if self.dropout:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros_like(h).index_add_(0, edge_dst, alpha.unsqueeze(1) * h_src)
        return F.relu(out)

# ---- Bi-GAT for bipartite topology view (direction-aware) ----

class BiGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, alpha: float = 0.2):
        super().__init__()
        self.W_drug = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.W_dis  = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.a_drug = nn.Parameter(torch.Tensor(2 * out_dim, 1))  
        self.a_dis  = nn.Parameter(torch.Tensor(2 * out_dim, 1)) 
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = dropout
        for p in [self.W_drug, self.W_dis, self.a_drug, self.a_dis]:
            nn.init.xavier_uniform_(p)
        self.last_alpha_d = None  
        self.last_alpha_p = None  

    def _agg_direction(self, src_h, dst_h, dst_idx, N, att_vec):
        e = self.leaky_relu(torch.matmul(torch.cat([dst_h, src_h], 1), att_vec)).squeeze(1)
        e = e - e.max()
        exp_e = torch.exp(e)
        denom = torch.zeros(N, device=src_h.device).index_add_(0, dst_idx, exp_e)
        alpha = exp_e / (denom[dst_idx] + 1e-16)
        if self.dropout:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def forward(self, drug_x, disease_x,
                dis_to_drug_src, dis_to_drug_dst,
                drug_to_dis_src, drug_to_dis_dst):
        n_drug, n_dis = drug_x.size(0), disease_x.size(0)
        h_drug = drug_x @ self.W_drug
        h_dis  = disease_x @ self.W_dis
        if self.dropout:
            h_drug = F.dropout(h_drug, p=self.dropout, training=self.training)
            h_dis  = F.dropout(h_dis,  p=self.dropout, training=self.training)

        # disease -> drug
        alpha_d = self._agg_direction(
            h_dis[dis_to_drug_src], h_drug[dis_to_drug_dst],
            dis_to_drug_dst, n_drug, self.a_drug)
        drug_out = torch.zeros_like(h_drug).index_add_(
            0, dis_to_drug_dst, alpha_d.unsqueeze(1) * h_dis[dis_to_drug_src])
        drug_out = F.relu(drug_out)

        # drug -> disease
        alpha_p = self._agg_direction(
            h_drug[drug_to_dis_src], h_dis[drug_to_dis_dst],
            drug_to_dis_dst, n_dis, self.a_dis)
        dis_out = torch.zeros_like(h_dis).index_add_(
            0, drug_to_dis_dst, alpha_p.unsqueeze(1) * h_drug[drug_to_dis_src])
        dis_out = F.relu(dis_out)

        # cache attention for analysis
        self.last_alpha_d = alpha_d.detach()
        self.last_alpha_p = alpha_p.detach()
        return drug_out, dis_out

# ---- Full BiGAT-Fusion ----

class BiGATFusionModel(nn.Module):
    def __init__(self,
                 n_drugs: int,
                 n_diseases: int,
                 drug_feat_neighbors: list,
                 disease_feat_neighbors: list,
                 drug_neighbors: dict,
                 disease_neighbors: dict,
                 *,
                 embed_dim: int = 64,
                 hidden_dim: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim

        self.drug_emb    = nn.Embedding(n_drugs, embed_dim)
        self.disease_emb = nn.Embedding(n_diseases, embed_dim)

        self.gat_drug_feat = GATLayer(embed_dim, embed_dim, dropout)
        self.gat_dis_feat  = GATLayer(embed_dim, embed_dim, dropout)

        self.bipartite_gat = BiGATLayer(embed_dim, embed_dim, dropout)

        self.gate_drug = nn.Linear(embed_dim * 2, 1)
        self.gate_dis  = nn.Linear(embed_dim * 2, 1)
        nn.init.constant_(self.gate_drug.bias, 0.0)
        nn.init.constant_(self.gate_dis.bias , 0.0)

        self.bias_d = nn.Embedding(n_drugs, 1)
        self.bias_p = nn.Embedding(n_diseases, 1)
        nn.init.zeros_(self.bias_d.weight); nn.init.zeros_(self.bias_p.weight)

        self.decoder = ResidualMoEDecoder(embed_dim, hidden_dim, dropout)

        self._register_edge_buffers(drug_feat_neighbors, disease_feat_neighbors,
                                    drug_neighbors, disease_neighbors)

    def _register_edge_buffers(self,
                               drug_feat_neighbors, disease_feat_neighbors,
                               drug_neighbors, disease_neighbors):
        import torch
        def make_edges(nei_list):
            src, dst = [], []
            for i, lst in enumerate(nei_list):
                for j in lst:
                    src.append(j)   
                    dst.append(i)   
            return torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)

        drug_feat_src,    drug_feat_dst    = make_edges(drug_feat_neighbors)
        disease_feat_src, disease_feat_dst = make_edges(disease_feat_neighbors)

        dis_to_drug_src, dis_to_drug_dst = [], []
        for d, p_list in drug_neighbors.items():
            for p in p_list:
                dis_to_drug_src.append(p)
                dis_to_drug_dst.append(d)
        drug_to_dis_src, drug_to_dis_dst = [], []
        for p, d_list in disease_neighbors.items():
            for d in d_list:
                drug_to_dis_src.append(d)
                drug_to_dis_dst.append(p)

        mapping = {
            'drug_feat_src': drug_feat_src, 'drug_feat_dst': drug_feat_dst,
            'disease_feat_src': disease_feat_src, 'disease_feat_dst': disease_feat_dst,
            'dis_to_drug_src': torch.tensor(dis_to_drug_src, dtype=torch.long),
            'dis_to_drug_dst': torch.tensor(dis_to_drug_dst, dtype=torch.long),
            'drug_to_dis_src': torch.tensor(drug_to_dis_src, dtype=torch.long),
            'drug_to_dis_dst': torch.tensor(drug_to_dis_dst, dtype=torch.long),
        }
        for name, tensor in mapping.items():
            self.register_buffer(name, tensor)

    def forward(self):
        drug_init = self.drug_emb.weight
        dis_init  = self.disease_emb.weight

        drug_feat = self.gat_drug_feat(drug_init, self.drug_feat_src, self.drug_feat_dst)
        dis_feat  = self.gat_dis_feat (dis_init , self.disease_feat_src, self.disease_feat_dst)

        drug_topo, dis_topo = self.bipartite_gat(
            drug_init, dis_init,
            self.dis_to_drug_src, self.dis_to_drug_dst,
            self.drug_to_dis_src, self.drug_to_dis_dst)

        gate_d = torch.sigmoid(self.gate_drug(torch.cat([drug_feat, drug_topo], 1)))
        gate_p = torch.sigmoid(self.gate_dis (torch.cat([dis_feat , dis_topo], 1)))
        drug_fused = gate_d * drug_feat + (1 - gate_d) * drug_topo
        dis_fused  = gate_p * dis_feat  + (1 - gate_p) * dis_topo

        return drug_fused, dis_fused

    def logits_on_pairs(self, drug_idx, dis_idx):
        drug_z, dis_z = self.forward()
        hd = drug_z[drug_idx]   # [B, D]
        hp = dis_z[dis_idx]     # [B, D]
        return self.decoder(hd, hp, drug_idx, dis_idx, self.bias_d, self.bias_p)

    def get_fused_embeddings(self):
        return self.forward()
