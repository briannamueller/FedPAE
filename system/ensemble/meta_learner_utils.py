"""GNN meta-learner architecture and utilities for FedDES."""
from __future__ import annotations

import torch
from torch import nn

from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv
from torch_geometric.utils import to_undirected


class HeteroGAT(nn.Module):
    """Heterogeneous graph attention meta-learner for FedDES.

    Uses PyG's GATv2Conv across two edge types:
      - sample-sample (ss): CMDW-based k-NN edges
      - classifier-sample (cs): top-k classifier competence edges

    Produces per-sample logits [N, M] — one logit per candidate classifier,
    representing predicted competence on that sample.
    """

    def __init__(
        self,
        metadata,
        input_dims,
        *,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        out_dim: int = 1,
        use_sample_residual: bool = False,
        use_edge_attr: bool = False,
    ) -> None:
        super().__init__()

        node_types, edge_types = metadata
        self.node_types = node_types
        self.edge_types = edge_types
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.use_sample_residual = bool(use_sample_residual)
        self.use_edge_attr = bool(use_edge_attr)

        self.input_proj = nn.ModuleDict(
            {ntype: nn.Linear(input_dims[ntype], hidden_dim) for ntype in node_types}
        )

        edge_dim = 1 if self.use_edge_attr else None
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                edge_type: GATv2Conv(
                    (hidden_dim, hidden_dim), hidden_dim,
                    heads=heads, concat=False, dropout=dropout,
                    add_self_loops=False, edge_dim=edge_dim,
                )
                for edge_type in edge_types
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        self.sample_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Returns logits [num_samples, out_dim]."""
        x_dict = {ntype: self.input_proj[ntype](data[ntype].x) for ntype in self.node_types}

        sample_residual = x_dict.get("sample") if self.use_sample_residual else None

        for layer_idx, conv in enumerate(self.convs):
            if self.use_edge_attr:
                edge_attr_dict = {}
                for et in data.edge_index_dict:
                    ea = getattr(data[et], "edge_attr", None)
                    if ea is not None:
                        edge_attr_dict[et] = ea.view(-1, 1) if ea.dim() == 1 else ea
                x_dict = conv(x_dict, data.edge_index_dict, edge_attr_dict)
            else:
                x_dict = conv(x_dict, data.edge_index_dict)

            if layer_idx < len(self.convs) - 1:
                x_dict = {ntype: self.dropout(self.activation(x)) for ntype, x in x_dict.items()}
            else:
                x_dict = {ntype: self.activation(x) for ntype, x in x_dict.items()}

        h = x_dict["sample"]
        if sample_residual is not None:
            h = h + sample_residual
        return self.sample_head(h)


def build_meta_learner(args, metadata, input_dims, num_candidates: int) -> HeteroGAT:
    """Construct the HeteroGAT meta-learner configured from ``args``."""
    return HeteroGAT(
        metadata=metadata,
        input_dims=dict(input_dims),
        hidden_dim=int(getattr(args, "gnn_hidden_dim", 128)),
        num_layers=int(getattr(args, "gnn_layers", 2)),
        heads=int(getattr(args, "gnn_heads", 4)),
        dropout=float(getattr(args, "gnn_dropout", 0.1)),
        out_dim=num_candidates,
        use_sample_residual=bool(getattr(args, "gnn_use_sample_residual", False)),
        use_edge_attr=bool(getattr(args, "gnn_use_edge_attr", False)),
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def compute_sample_weights(client, labels: torch.Tensor, meta_labels: torch.Tensor, mode: str):
    """Per-sample weights for GNN training loss.

    mode: ``"class_prevalence"`` (inverse class frequency),
          ``"difficulty"`` (1 - fraction correct), or ``"none"``/``""``.
    """
    mode = str(mode).lower()
    if mode == "class_prevalence":
        counts = torch.bincount(labels.detach().cpu(), minlength=client.args.num_classes).float()
        counts = counts.to(labels.device).clamp(min=1.0)
        return ((counts.sum() / counts.numel()) / counts)[labels]
    if mode == "difficulty":
        return (1.0 - meta_labels.float().mean(dim=1)).clamp(min=0.0)
    return None


def enforce_bidirectionality(data: HeteroData, bidirectional: bool) -> HeteroData:
    """Make sample-sample and classifier-sample edges bidirectional for train nodes only.

    Val/test nodes remain receive-only so their labels don't leak into training.
    """
    if not bidirectional:
        return data

    def _num_nodes(ntype):
        n = getattr(data[ntype], "num_nodes", None)
        if n is not None:
            return int(n)
        x = getattr(data[ntype], "x", None)
        return int(x.size(0)) if x is not None else 0

    # Sample-Sample: make undirected, keep only train-source edges
    rel = ("sample", "ss", "sample")
    if rel in data.edge_index_dict:
        ei = data[rel].edge_index
        eattr = getattr(data[rel], "edge_attr", None)
        if eattr is not None:
            ei_ud, eattr_ud = to_undirected(ei, eattr, num_nodes=_num_nodes("sample"))
        else:
            ei_ud = to_undirected(ei, num_nodes=_num_nodes("sample"))
            eattr_ud = None
        train_mask = getattr(data["sample"], "train_mask", None)
        if train_mask is not None:
            keep = train_mask.bool()[ei_ud[0]]
            data[rel].edge_index = ei_ud[:, keep]
            if eattr_ud is not None:
                data[rel].edge_attr = eattr_ud[keep]
        else:
            data[rel].edge_index = ei_ud[:, :0]

    # Classifier->Sample: add reverse edges for train samples only
    cs_rel = ("classifier", "cs", "sample")
    sc_rel = ("sample", "cs_rev", "classifier")
    if cs_rel in data.edge_index_dict:
        cs_ei = data[cs_rel].edge_index
        cs_eattr = getattr(data[cs_rel], "edge_attr", None)
        train_mask = getattr(data["sample"], "train_mask", None)
        if train_mask is not None:
            keep = train_mask.bool()[cs_ei[1]]
            if keep.any():
                data[sc_rel].edge_index = torch.stack([cs_ei[1, keep], cs_ei[0, keep]], dim=0)
                if cs_eattr is not None:
                    data[sc_rel].edge_attr = cs_eattr[keep]

    return data
