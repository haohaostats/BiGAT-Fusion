
#!/usr/bin/env python3
# run_fullcv.py  —  10x10 CV training/evaluation for BiGAT-Fusion model.

import argparse, csv, random, os, json
import numpy as np
import torch as th
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from pathlib import Path

from data_loader          import load_dataset
from bigat_model          import BiGATFusionModel

# ---- Utilities ----

class PosDataset(Dataset):
    """Iterate over positive edges; for each, sample neg_k unknown pairs from the global pool."""
    def __init__(self, pos_edges, n_d, n_p, neg_k=1, pos_set=None, rng=None):
        self.pos           = pos_edges
        self.n_d, self.n_p = n_d, n_p
        self.pos_set       = set(pos_edges) if pos_set is None else set(pos_set)
        self.neg_k         = neg_k
        self.rng           = random.Random(0) if rng is None else rng

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        d, p = self.pos[idx]
        negs = []
        while len(negs) < self.neg_k:
            dn = self.rng.randrange(self.n_d)
            pn = self.rng.randrange(self.n_p)
            if (dn, pn) not in self.pos_set:
                negs.append((dn, pn, 0))
        return (d, p, 1), negs

def collate(batch):
    d_idx, p_idx, lbl = [], [], []
    for pos, negs in batch:
        for d, p, y in (pos, *negs):
            d_idx.append(d); p_idx.append(p); lbl.append(y)
    return (th.tensor(d_idx),
            th.tensor(p_idx),
            th.tensor(lbl, dtype=th.float32))

@th.no_grad()
def evaluate(model, edges, pos_set, device):
    model.eval()
    d_idx = th.tensor([e[0] for e in edges], device=device)
    p_idx = th.tensor([e[1] for e in edges], device=device)
    logits  = model.logits_on_pairs(d_idx, p_idx)
    y_score = th.sigmoid(logits).cpu().numpy()
    y_true  = np.fromiter((1 if e in pos_set else 0 for e in edges), int)

    fpr, tpr,  _ = metrics.roc_curve(y_true, y_score)
    auc_roc = metrics.auc(fpr, tpr)

    prec, rec, _ = metrics.precision_recall_curve(y_true, y_score)
    auc_pr  = metrics.auc(rec, prec)
    return auc_roc, auc_pr

def build_folds(all_pairs, k, seed):
    rng = random.Random(seed)
    pairs = all_pairs[:]
    rng.shuffle(pairs)
    return [pairs[i::k] for i in range(k)]

def run_once(args):
    device = th.device("cuda" if args.device == "cuda" and th.cuda.is_available() else "cpu")

    base     = load_dataset(args.mat_path, k=args.k, val_ratio=0., test_ratio=0.)
    n_d, n_p = base['n_drugs'], base['n_diseases']
    pos_set  = base['assoc_pos_set']
    all_pairs = [(i, j) for i in range(n_d) for j in range(n_p)]

    ds_name = Path(args.mat_path).stem
    out_dir = Path("results") / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logs = []
    rng_master = random.Random(args.seed)

    for rpt in range(args.repeats):
        folds = build_folds(all_pairs, args.folds, rng_master.randint(0, 1 << 30))

        for fid in range(args.folds):
            test_edges = folds[fid]
            val_id     = (fid + 1) % args.folds
            val_edges  = folds[val_id]

            train_pos  = [e for k, f in enumerate(folds) if k not in (fid, val_id) for e in f if e in pos_set]

            drug_adj = {i: [] for i in range(n_d)}
            dis_adj  = {j: [] for j in range(n_p)}
            for d, p in train_pos:
                drug_adj[d].append(p)
                dis_adj[p].append(d)

            model = BiGATFusionModel(
                n_drugs=n_d,
                n_diseases=n_p,
                drug_feat_neighbors=base['drug_feat_neighbors'],
                disease_feat_neighbors=base['disease_feat_neighbors'],
                drug_neighbors=drug_adj,
                disease_neighbors=dis_adj,
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout
            ).to(device)

            gate_params, backbone_params = [], []
            for name, param in model.named_parameters():
                if not param.requires_grad: continue
                if "gate_" in name:
                    gate_params.append(param)
                else:
                    backbone_params.append(param)

            param_groups = [{"params": backbone_params, "weight_decay": args.wd_backbone}]
            if gate_params:
                param_groups.append({"params": gate_params, "weight_decay": args.wd_gate})

            opt   = Adam(param_groups, lr=args.lr)
            sched = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=args.lr_patience,
                                      threshold=1e-4, min_lr=1e-5)
            criterion = nn.BCEWithLogitsLoss()

            loader = DataLoader(
                PosDataset(train_pos, n_d, n_p, args.neg_k,
                           pos_set=pos_set, rng=random.Random(args.seed + rpt*1000 + fid)),
                batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                num_workers=0, drop_last=False
            )

            best_val = -1.0
            for ep in range(1, args.epochs + 1):
                model.train()
                for d_idx, p_idx, lbl in loader:
                    d_idx, p_idx, lbl = (d_idx.to(device), p_idx.to(device), lbl.to(device))
                    logits = model.logits_on_pairs(d_idx, p_idx)
                    loss   = criterion(logits, lbl)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()

                if (ep % args.eval_every == 0) or (ep == args.epochs):
                    auc_val, _ = evaluate(model, val_edges, pos_set, device)
                    sched.step(auc_val)
                    if auc_val > best_val + 1e-6:
                        best_val = auc_val
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if 'best_state' in locals():
                model.load_state_dict(best_state, strict=False)

            auc, aupr = evaluate(model, test_edges, pos_set, device)
            logs.append({'Ablation': 'full', 'Repeat': rpt, 'Fold': fid, 'ValFold': val_id, 'AUROC': auc, 'AUPRC': aupr})

            ck_dir = Path("train"); ck_dir.mkdir(parents=True, exist_ok=True)
            ckpt_name = f"full_{ds_name}_r{rpt}_f{fid}.pt"
            th.save(model.state_dict(), ck_dir / ckpt_name)

            from pathlib import Path as _P
            import numpy as _np
            with th.no_grad():
                model.eval()
                drug_init = model.drug_emb.weight
                dis_init  = model.disease_emb.weight

                drug_feat = model.gat_drug_feat(drug_init, model.drug_feat_src, model.drug_feat_dst)
                dis_feat  = model.gat_dis_feat (dis_init,   model.disease_feat_src, model.disease_feat_dst)
                drug_topo, dis_topo = model.bipartite_gat(
                    drug_init, dis_init, model.dis_to_drug_src, model.dis_to_drug_dst,
                    model.drug_to_dis_src, model.drug_to_dis_dst
                )

                alpha_d = model.bipartite_gat.last_alpha_d.detach().cpu().numpy()  # disease->drug
                alpha_p = model.bipartite_gat.last_alpha_p.detach().cpu().numpy()  # drug->disease
                src_d = model.dis_to_drug_src.cpu().numpy(); dst_d = model.dis_to_drug_dst.cpu().numpy()
                src_p = model.drug_to_dis_src.cpu().numpy(); dst_p = model.drug_to_dis_dst.cpu().numpy()

                g_d = th.sigmoid(model.gate_drug(th.cat([drug_feat, drug_topo], 1))).squeeze(1).cpu().numpy()
                g_p = th.sigmoid(model.gate_dis (th.cat([dis_feat , dis_topo], 1))).squeeze(1).cpu().numpy()

                intp_dir = _P("interpret") / "folds" / ds_name / "full"
                intp_dir.mkdir(parents=True, exist_ok=True)
                meta_json = json.dumps({"dataset": ds_name, "repeat": int(rpt), "fold": int(fid), "val_fold": int(val_id)})
                _np.savez_compressed(intp_dir / f"r{rpt}_f{fid}.npz",
                                     g_d=g_d, g_p=g_p,
                                     alpha_d=alpha_d, src_d=src_d, dst_d=dst_d,
                                     alpha_p=alpha_p, src_p=src_p, dst_p=dst_p,
                                     meta=meta_json)

            print(f"[{ds_name}|full] R{rpt+1}/{args.repeats} F{fid+1}/{args.folds} (val={val_id}) AUROC={auc:.4f} AUPRC={aupr:.4f}")

    csv_path = out_dir / f"cv_log_full.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=logs[0].keys())
        writer.writeheader(); writer.writerows(logs)
    print(f"==> Saved {csv_path}")

    au = np.array([l['AUROC'] for l in logs])
    pr = np.array([l['AUPRC'] for l in logs])
    print(f"[{ds_name}|full] AUROC={au.mean():.4f}±{au.std():.4f} AUPRC={pr.mean():.4f}±{pr.std():.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_path", required=True, help="C/G .mat | Ldataset dir | LRSSL dir")

    parser.add_argument("--embed_dim",   type=int,   default=128)
    parser.add_argument("--hidden_dim",  type=int,   default=256)
    parser.add_argument("--k",           type=int,   default=4,    help="k-NN k for feature graph")
    parser.add_argument("--epochs",      type=int,   default=4000)
    parser.add_argument("--lr",          type=float, default=0.001)
    parser.add_argument("--batch_size",  type=int,   default=1024)
    parser.add_argument("--neg_k",       type=int,   default=3,    help="negatives per positive")
    parser.add_argument("--folds",       type=int,   default=10)
    parser.add_argument("--repeats",     type=int,   default=10)
    parser.add_argument("--seed",        type=int,   default=2025)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--device",      choices=["cpu", "cuda"], default="cuda")

    parser.add_argument("--wd_backbone", type=float, default=1e-4, help="weight_decay for non-gate parameters")
    parser.add_argument("--wd_gate",     type=float, default=1e-3, help="weight_decay for gate parameters")

    parser.add_argument("--eval_every",  type=int,   default=20,   help="evaluate on validation fold every N epochs")
    parser.add_argument("--lr_patience", type=int,   default=20,   help="ReduceLROnPlateau patience (in eval steps)")
    parser.add_argument("--early_stop",  action="store_true",      help="enable early stopping on validation AUROC")
    parser.add_argument("--es_patience", type=int,   default=40,   help="early stopping patience (in eval steps)")

    args = parser.parse_args()

    random.seed(args.seed); th.manual_seed(args.seed); np.random.seed(args.seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    run_once(args)

if __name__ == "__main__":
    main()
