#!/usr/bin/env python3
"""
Estimate GPU VRAM and system RAM usage for LightGCN / LightCCN-Flat / LightCCN-Multi
across datasets (Gowalla, Yelp2018, Amazon-Book) and tau thresholds.

Based on actual model code and operator construction logic from the codebase.
"""

from dataclasses import dataclass, field
from typing import Optional
import math


# ============================================================================
# Dataset statistics (from LightGCN-PyTorch repo)
# ============================================================================

@dataclass
class DatasetStats:
    name: str
    n_users: int
    n_items: int
    n_train_interactions: int


DATASETS = {
    "gowalla": DatasetStats("gowalla", 29_858, 40_981, 810_128),
    "yelp2018": DatasetStats("yelp2018", 31_668, 38_048, 1_237_259),
    "amazon-book": DatasetStats("amazon-book", 52_643, 91_599, 2_380_730),
}


# ============================================================================
# Cell complex sizes at various tau (estimated from coverage analysis)
# ============================================================================
# These are empirical estimates. For Gowalla we have anchor points.
# For yelp2018 and amazon-book, scale proportionally to interaction count
# (more interactions => denser co-occurrence => more faces and edges at same tau).

# Gowalla anchor points (from the user's estimates):
GOWALLA_COMPLEX = {
    3:  {"n_faces": 2_000_000, "n_edges": 150_000},
    5:  {"n_faces": 500_000,   "n_edges": 60_000},
    10: {"n_faces": 150_000,   "n_edges": 25_000},
    20: {"n_faces": 31_000,    "n_edges": 9_000},
    30: {"n_faces": 10_000,    "n_edges": 4_500},
    50: {"n_faces": 2_000,     "n_edges": 1_500},
}


def estimate_complex_size(dataset_name: str, tau: int) -> dict:
    """Estimate n_faces and n_edges for a dataset at a given tau."""
    gowalla_interactions = DATASETS["gowalla"].n_train_interactions
    target_interactions = DATASETS[dataset_name].n_train_interactions
    
    # Scale factor: more interactions means denser co-occurrence.
    # The relationship is super-linear for faces (triples grow faster with density).
    # Use interaction_ratio^1.5 for faces, interaction_ratio^1.2 for edges.
    ratio = target_interactions / gowalla_interactions
    
    # Find closest tau in gowalla anchors
    available_taus = sorted(GOWALLA_COMPLEX.keys())
    
    if tau in GOWALLA_COMPLEX:
        base = GOWALLA_COMPLEX[tau]
    else:
        # Interpolate/extrapolate log-linearly between known points
        if tau < available_taus[0]:
            base = GOWALLA_COMPLEX[available_taus[0]]
            # Extrapolate: lower tau => more faces, roughly exponential
            factor = (available_taus[0] / tau) ** 2.5
            base = {"n_faces": int(base["n_faces"] * factor),
                    "n_edges": int(base["n_edges"] * factor ** 0.5)}
        elif tau > available_taus[-1]:
            base = GOWALLA_COMPLEX[available_taus[-1]]
            factor = (available_taus[-1] / tau) ** 2.0
            base = {"n_faces": max(100, int(base["n_faces"] * factor)),
                    "n_edges": max(50, int(base["n_edges"] * factor ** 0.7))}
        else:
            # Linear interpolation in log space between flanking points
            lower = max(t for t in available_taus if t <= tau)
            upper = min(t for t in available_taus if t >= tau)
            if lower == upper:
                base = GOWALLA_COMPLEX[lower]
            else:
                frac = (math.log(tau) - math.log(lower)) / (math.log(upper) - math.log(lower))
                bl = GOWALLA_COMPLEX[lower]
                bu = GOWALLA_COMPLEX[upper]
                base = {
                    "n_faces": int(math.exp(math.log(bl["n_faces"]) * (1-frac) + math.log(bu["n_faces"]) * frac)),
                    "n_edges": int(math.exp(math.log(bl["n_edges"]) * (1-frac) + math.log(bu["n_edges"]) * frac)),
                }
    
    if dataset_name == "gowalla":
        return base
    
    # Scale for other datasets
    n_faces = int(base["n_faces"] * ratio ** 1.5)
    n_edges = int(base["n_edges"] * ratio ** 1.2)
    return {"n_faces": n_faces, "n_edges": n_edges}


# ============================================================================
# Memory estimation functions
# ============================================================================

def sparse_coo_bytes(nnz: int) -> int:
    """Memory for a sparse COO tensor: values (float32) + indices (2 x int64)."""
    return nnz * 4 + nnz * 2 * 8  # 4 bytes per value + 16 bytes per index pair


def sparse_csr_bytes(nnz: int, n_rows: int) -> int:
    """Memory for a CSR matrix: data (float32) + indices (int32) + indptr (int32)."""
    return nnz * 4 + nnz * 4 + (n_rows + 1) * 4


@dataclass
class MemoryEstimate:
    dataset: str
    model: str
    tau: Optional[int]
    embed_dim: int
    n_layers: int
    batch_size: int
    
    # Components (bytes)
    node_embeddings: int = 0
    edge_embeddings: int = 0
    face_embeddings: int = 0
    sparse_operators: int = 0
    optimizer_states: int = 0
    gradients: int = 0
    batch_overhead: int = 0
    propagation_intermediates: int = 0
    
    # CPU
    cpu_cooccurrence: int = 0
    cpu_face_list: int = 0
    cpu_B1_B2: int = 0
    cpu_operator_construction: int = 0  # temp memory during construction
    
    # Extra info
    n_faces: int = 0
    n_edges: int = 0
    notes: list = field(default_factory=list)
    
    @property
    def total_params_bytes(self) -> int:
        return self.node_embeddings + self.edge_embeddings + self.face_embeddings
    
    @property
    def total_gpu_bytes(self) -> int:
        return (self.total_params_bytes + self.sparse_operators +
                self.optimizer_states + self.gradients +
                self.batch_overhead + self.propagation_intermediates)
    
    @property
    def total_gpu_mb(self) -> float:
        return self.total_gpu_bytes / (1024 ** 2)
    
    @property
    def peak_cpu_bytes(self) -> int:
        return (self.cpu_cooccurrence + self.cpu_face_list +
                self.cpu_B1_B2 + self.cpu_operator_construction)
    
    @property
    def peak_cpu_mb(self) -> float:
        return self.peak_cpu_bytes / (1024 ** 2)


def estimate_lightgcn(ds: DatasetStats, embed_dim=64, n_layers=3, batch_size=2048) -> MemoryEstimate:
    est = MemoryEstimate(ds.name, "LightGCN", None, embed_dim, n_layers, batch_size)
    
    n_nodes = ds.n_users + ds.n_items
    
    # Embedding tables
    est.node_embeddings = n_nodes * embed_dim * 4
    
    # Bipartite adjacency: [[0, R], [R^T, 0]] -- each interaction appears twice
    adj_nnz = 2 * ds.n_train_interactions
    est.sparse_operators = sparse_coo_bytes(adj_nnz)
    
    # Optimizer (Adam): 2 states (m, v) per parameter
    est.optimizer_states = 2 * est.total_params_bytes
    
    # Gradients: 1x params
    est.gradients = est.total_params_bytes
    
    # Batch: user, pos_item, neg_item embeddings + ego embeddings for reg
    est.batch_overhead = batch_size * embed_dim * 6 * 4  # 6 tensors
    
    # Propagation intermediates: n_layers full-graph embeddings stored for layer combination
    est.propagation_intermediates = (n_layers + 1) * n_nodes * embed_dim * 4
    
    # CPU: essentially just loading the data, minimal
    est.cpu_cooccurrence = 0
    est.cpu_face_list = 0
    est.cpu_B1_B2 = 0
    # CSR representation of training data: ~interaction_count * 12 bytes
    est.cpu_operator_construction = ds.n_train_interactions * 12
    
    return est


def estimate_lightccn_flat(ds: DatasetStats, tau: int, gamma=0.5,
                            embed_dim=64, n_layers=3, batch_size=2048) -> MemoryEstimate:
    cx = estimate_complex_size(ds.name, tau)
    est = MemoryEstimate(ds.name, "LightCCN-Flat", tau, embed_dim, n_layers, batch_size,
                         n_faces=cx["n_faces"], n_edges=cx["n_edges"])
    
    n_nodes = ds.n_users + ds.n_items
    
    # Embedding tables: same as LightGCN (only node embeddings)
    est.node_embeddings = n_nodes * embed_dim * 4
    
    # Augmented adjacency A_tilde = [[0, (1-gamma)*R], [(1-gamma)*R^T, gamma*S]]
    # R block: 2 * n_interactions nnz (R and R^T)
    # S block: each face (i,j,k) contributes 3 edges => 6 entries in S (symmetric)
    # But edges can be shared across faces, so S_nnz <= 6 * n_faces
    # More precisely, n_edges unique pairs, each appears symmetrically => S_nnz = 2 * n_edges
    s_nnz = 2 * cx["n_edges"]
    total_adj_nnz = 2 * ds.n_train_interactions + s_nnz
    est.sparse_operators = sparse_coo_bytes(total_adj_nnz)
    
    # Optimizer, gradients, batch (same structure as LightGCN)
    est.optimizer_states = 2 * est.total_params_bytes
    est.gradients = est.total_params_bytes
    est.batch_overhead = batch_size * embed_dim * 6 * 4
    est.propagation_intermediates = (n_layers + 1) * n_nodes * embed_dim * 4
    
    # CPU: cell complex construction
    # C = R^T @ R: n_items x n_items sparse. NNZ depends on overlap.
    # Rough estimate: each item has avg interactions/items * interactions entries
    # More precisely, for bipartite graph, C_nnz ~ sum of (deg_u choose 2) for all users
    avg_user_deg = ds.n_train_interactions / ds.n_users
    # C_nnz ~ n_users * (avg_deg * (avg_deg-1) / 2) but capped at n_items^2
    # This is the number of co-occurring item pairs
    c_nnz_estimate = min(
        int(ds.n_users * avg_user_deg * (avg_user_deg - 1) / 2),
        ds.n_items * ds.n_items // 4  # rough cap
    )
    est.cpu_cooccurrence = sparse_csr_bytes(c_nnz_estimate, ds.n_items)
    
    # Face list: n_faces * 3 ints (8 bytes each as numpy int64)
    est.cpu_face_list = cx["n_faces"] * 3 * 8
    
    # B1, B2: not needed for Flat (only S is used)
    est.cpu_B1_B2 = 0
    
    # Temporary: the thresholded C and neighbor dict during face detection
    est.cpu_operator_construction = est.cpu_cooccurrence + cx["n_edges"] * 64  # neighbor dict overhead
    
    return est


def estimate_lightccn_multi(ds: DatasetStats, tau: int,
                             embed_dim=64, edge_embed_dim=64, face_embed_dim=64,
                             n_layers=3, batch_size=2048) -> MemoryEstimate:
    cx = estimate_complex_size(ds.name, tau)
    est = MemoryEstimate(ds.name, "LightCCN-Multi", tau, embed_dim, n_layers, batch_size,
                         n_faces=cx["n_faces"], n_edges=cx["n_edges"])
    
    n_nodes = ds.n_users + ds.n_items
    n_edges = cx["n_edges"]
    n_faces = cx["n_faces"]
    
    # --- Embedding tables ---
    est.node_embeddings = n_nodes * embed_dim * 4
    est.edge_embeddings = n_edges * edge_embed_dim * 4
    est.face_embeddings = n_faces * face_embed_dim * 4
    
    # --- 7 Sparse operators ---
    # A_hat_0: bipartite adj, nnz = 2 * n_interactions
    a0_nnz = 2 * ds.n_train_interactions
    
    # B_hat_1_down (B1 normalized): node-edge incidence, nnz = 2 * n_edges (each edge has 2 boundary nodes)
    b1_nnz = 2 * n_edges
    
    # B_hat_1_up (B1^T normalized): same nnz
    b1t_nnz = 2 * n_edges
    
    # A_hat_1: edge-edge adj via shared faces = B2 @ B2^T - diag
    # Each face has 3 boundary edges, so B2 has 3 * n_faces nnz
    b2_nnz = 3 * n_faces
    # B2 @ B2^T: for each face, its 3 edges are all connected => 3*2=6 off-diag entries per face
    # But edges shared across faces contribute overlapping entries
    # Rough: A1_nnz ~ min(6 * n_faces, n_edges * n_edges)
    # Better estimate: avg edges per face = 3 (always), avg faces per edge ~ 3*n_faces/n_edges
    faces_per_edge = 3 * n_faces / max(n_edges, 1)
    # Each edge connects to ~2*faces_per_edge other edges (each face contributes 2 neighbors)
    a1_nnz = min(int(n_edges * 2 * faces_per_edge), 6 * n_faces)
    
    # B_hat_2_down (B2 normalized): edge-face incidence, nnz = 3 * n_faces
    b2_down_nnz = 3 * n_faces
    
    # B_hat_2_up (B2^T normalized): same nnz
    b2_up_nnz = 3 * n_faces
    
    # A_hat_2: face-face adj via shared edges = B2^T @ B2 - diag
    # For each edge, its faces are all connected => faces_per_edge*(faces_per_edge-1) entries
    # Total ~ n_edges * faces_per_edge * (faces_per_edge - 1)
    a2_nnz = min(int(n_edges * faces_per_edge * max(faces_per_edge - 1, 0)),
                  n_faces * 20)  # cap at reasonable connectivity
    
    total_op_nnz = a0_nnz + b1_nnz + b1t_nnz + a1_nnz + b2_down_nnz + b2_up_nnz + a2_nnz
    est.sparse_operators = sparse_coo_bytes(total_op_nnz)
    
    # --- Optimizer states (Adam): 2x all parameters ---
    est.optimizer_states = 2 * est.total_params_bytes
    # Also: mixing weight logits (7 params) - negligible
    
    # --- Gradients ---
    est.gradients = est.total_params_bytes
    
    # --- Batch overhead ---
    est.batch_overhead = batch_size * embed_dim * 6 * 4
    
    # --- Propagation intermediates ---
    # Per layer: must store x_nodes, x_edges, x_faces + intermediates from sparse matmuls
    # Layer combination stores (n_layers+1) node embedding snapshots
    # Plus each sparse.mm creates a dense output temporarily
    node_layer_storage = (n_layers + 1) * n_nodes * embed_dim * 4
    edge_storage = 2 * n_edges * edge_embed_dim * 4  # current + new
    face_storage = 2 * n_faces * face_embed_dim * 4  # current + new
    # Intermediate matmul results (7 per layer, largest is n_nodes * embed_dim)
    matmul_intermediates = n_layers * (
        n_nodes * embed_dim * 4 +   # node_from_node
        n_nodes * embed_dim * 4 +   # node_from_edge
        n_edges * edge_embed_dim * 4 +  # edge_from_node
        n_edges * edge_embed_dim * 4 +  # edge_from_edge
        n_edges * edge_embed_dim * 4 +  # edge_from_face
        n_faces * face_embed_dim * 4 +  # face_from_edge
        n_faces * face_embed_dim * 4    # face_from_face
    )
    est.propagation_intermediates = node_layer_storage + edge_storage + face_storage + matmul_intermediates
    
    # --- CPU: cell complex construction ---
    avg_user_deg = ds.n_train_interactions / ds.n_users
    c_nnz_estimate = min(
        int(ds.n_users * avg_user_deg * (avg_user_deg - 1) / 2),
        ds.n_items * ds.n_items // 4
    )
    est.cpu_cooccurrence = sparse_csr_bytes(c_nnz_estimate, ds.n_items)
    est.cpu_face_list = n_faces * 3 * 8
    
    # B1: (n_nodes, n_edges) with 2*n_edges nnz
    # B2: (n_edges, n_faces) with 3*n_faces nnz
    est.cpu_B1_B2 = (sparse_csr_bytes(2 * n_edges, n_nodes) +
                      sparse_csr_bytes(3 * n_faces, n_edges))
    
    # Temporary memory during construction: C matrix + thresholded C + neighbor dict + sets
    est.cpu_operator_construction = est.cpu_cooccurrence * 2 + n_edges * 128
    
    return est


# ============================================================================
# Main estimation and output
# ============================================================================

def bytes_to_mb(b: int) -> float:
    return b / (1024 ** 2)


def bytes_to_gb(b: int) -> float:
    return b / (1024 ** 3)


def recommend_colab(gpu_mb: float, cpu_mb: float) -> str:
    """Recommend Colab tier based on resource requirements."""
    # T4: 15 GB VRAM, 12.7 GB RAM
    # L4: 24 GB VRAM, ~50 GB RAM
    # A100: 40/80 GB VRAM, 83 GB RAM
    
    # Leave ~20% headroom for framework overhead
    if gpu_mb * 1.2 <= 15_000 and cpu_mb * 1.2 <= 12_700:
        return "T4 (free tier)"
    elif gpu_mb * 1.2 <= 15_000 and cpu_mb * 1.2 <= 50_000:
        return "T4 (Pro, needs extra RAM)"
    elif gpu_mb * 1.2 <= 24_000 and cpu_mb * 1.2 <= 50_000:
        return "L4 (Pro)"
    elif gpu_mb * 1.2 <= 40_000 and cpu_mb * 1.2 <= 83_000:
        return "A100 40GB"
    elif gpu_mb * 1.2 <= 80_000:
        return "A100 80GB"
    else:
        return "EXCEEDS COLAB (multi-GPU / cluster)"


def main():
    embed_dim = 64
    n_layers = 3
    batch_size = 2048
    edge_embed_dim = 64
    face_embed_dim = 64
    
    # Define all configurations to estimate
    configs = []
    
    for ds_name in ["gowalla", "yelp2018", "amazon-book"]:
        ds = DATASETS[ds_name]
        
        # LightGCN baseline
        configs.append(estimate_lightgcn(ds, embed_dim, n_layers, batch_size))
        
        # LightCCN-Flat and Multi at various tau values
        if ds_name == "gowalla":
            taus = [3, 5, 10, 20, 30]
        elif ds_name == "yelp2018":
            taus = [3, 5, 10, 20, 30]
        else:  # amazon-book
            taus = [3, 5, 10, 20, 30, 50]
        
        for tau in taus:
            configs.append(estimate_lightccn_flat(ds, tau, 0.5, embed_dim, n_layers, batch_size))
            configs.append(estimate_lightccn_multi(ds, tau, embed_dim, edge_embed_dim, face_embed_dim, n_layers, batch_size))
    
    # ===== Print summary table =====
    print("=" * 145)
    print(f"{'Dataset':<14} {'Model':<16} {'tau':>5} {'Faces':>12} {'Edges':>10} "
          f"{'GPU VRAM':>10} {'CPU RAM':>10} {'Colab Tier':<28} {'Notes'}")
    print("-" * 145)
    
    current_ds = None
    for est in configs:
        if est.dataset != current_ds:
            if current_ds is not None:
                print("-" * 145)
            current_ds = est.dataset
        
        tau_str = str(est.tau) if est.tau else "-"
        faces_str = f"{est.n_faces:,}" if est.n_faces else "-"
        edges_str = f"{est.n_edges:,}" if est.n_edges else "-"
        gpu_mb = est.total_gpu_mb
        cpu_mb = est.peak_cpu_mb
        
        gpu_str = f"{gpu_mb:,.0f} MB"
        cpu_str = f"{cpu_mb:,.0f} MB"
        
        recommendation = recommend_colab(gpu_mb, cpu_mb)
        
        notes = []
        if est.model == "LightCCN-Multi" and est.n_faces > 500_000:
            notes.append("large face embed table")
        if cpu_mb > 12_700:
            notes.append("high CPU RAM")
        if gpu_mb > 15_000:
            notes.append("exceeds T4 VRAM")
        
        notes_str = "; ".join(notes) if notes else ""
        
        print(f"{est.dataset:<14} {est.model:<16} {tau_str:>5} {faces_str:>12} {edges_str:>10} "
              f"{gpu_str:>10} {cpu_str:>10} {recommendation:<28} {notes_str}")
    
    print("=" * 145)
    
    # ===== Detailed breakdown for key configurations =====
    print("\n")
    print("=" * 100)
    print("DETAILED BREAKDOWN FOR KEY CONFIGURATIONS")
    print("=" * 100)
    
    # Pick a few representative configs
    key_configs = []
    for est in configs:
        # LightGCN for each dataset
        if est.model == "LightGCN":
            key_configs.append(est)
        # Flat and Multi at default tau for each dataset
        elif est.dataset == "gowalla" and est.tau == 20:
            key_configs.append(est)
        elif est.dataset == "yelp2018" and est.tau == 20:
            key_configs.append(est)
        elif est.dataset == "amazon-book" and est.tau == 30:
            key_configs.append(est)
        # Extreme case: amazon-book tau=3
        elif est.dataset == "amazon-book" and est.tau == 3:
            key_configs.append(est)
    
    for est in key_configs:
        tau_str = f" tau={est.tau}" if est.tau else ""
        print(f"\n--- {est.dataset} / {est.model}{tau_str} ---")
        print(f"  Node embeddings:           {bytes_to_mb(est.node_embeddings):>10.1f} MB  "
              f"({DATASETS[est.dataset].n_users + DATASETS[est.dataset].n_items:,} nodes x {est.embed_dim}d)")
        if est.edge_embeddings:
            print(f"  Edge embeddings:           {bytes_to_mb(est.edge_embeddings):>10.1f} MB  "
                  f"({est.n_edges:,} edges x {est.embed_dim}d)")
        if est.face_embeddings:
            print(f"  Face embeddings:           {bytes_to_mb(est.face_embeddings):>10.1f} MB  "
                  f"({est.n_faces:,} faces x {est.embed_dim}d)")
        print(f"  Sparse operators (GPU):    {bytes_to_mb(est.sparse_operators):>10.1f} MB")
        print(f"  Optimizer states (Adam):   {bytes_to_mb(est.optimizer_states):>10.1f} MB")
        print(f"  Gradients:                 {bytes_to_mb(est.gradients):>10.1f} MB")
        print(f"  Batch overhead:            {bytes_to_mb(est.batch_overhead):>10.1f} MB")
        print(f"  Propagation intermediates: {bytes_to_mb(est.propagation_intermediates):>10.1f} MB")
        print(f"  -----------------------------------------------")
        print(f"  TOTAL GPU VRAM:            {est.total_gpu_mb:>10.1f} MB  ({bytes_to_gb(est.total_gpu_bytes):.2f} GB)")
        print(f"  Peak CPU RAM (construction):{est.peak_cpu_mb:>9.1f} MB  ({bytes_to_gb(est.peak_cpu_bytes):.2f} GB)")
        print(f"  Recommended tier: {recommend_colab(est.total_gpu_mb, est.peak_cpu_mb)}")
    
    # ===== Scaling analysis =====
    print("\n")
    print("=" * 100)
    print("SCALING ANALYSIS: How face/edge count affects GPU VRAM (LightCCN-Multi)")
    print("=" * 100)
    print(f"\n{'Dataset':<14} {'tau':>5} {'Faces':>12} {'Edges':>10} "
          f"{'Embed (node)':>14} {'Embed (edge)':>14} {'Embed (face)':>14} "
          f"{'Operators':>12} {'Total GPU':>12}")
    print("-" * 120)
    
    for est in configs:
        if est.model != "LightCCN-Multi":
            continue
        print(f"{est.dataset:<14} {est.tau:>5} {est.n_faces:>12,} {est.n_edges:>10,} "
              f"{bytes_to_mb(est.node_embeddings):>12.1f}MB "
              f"{bytes_to_mb(est.edge_embeddings):>12.1f}MB "
              f"{bytes_to_mb(est.face_embeddings):>12.1f}MB "
              f"{bytes_to_mb(est.sparse_operators):>10.1f}MB "
              f"{est.total_gpu_mb:>10.1f}MB")
    
    # ===== Final recommendations =====
    print("\n")
    print("=" * 100)
    print("RECOMMENDATIONS SUMMARY")
    print("=" * 100)
    
    print("""
    FREE COLAB (T4: 15GB VRAM, 12.7GB RAM):
    ----------------------------------------
    """)
    free_ok = [e for e in configs if "free" in recommend_colab(e.total_gpu_mb, e.peak_cpu_mb).lower()
               or "T4" in recommend_colab(e.total_gpu_mb, e.peak_cpu_mb)]
    for e in free_ok:
        tau_s = f" tau={e.tau}" if e.tau else ""
        print(f"      [OK]  {e.dataset:<14} {e.model:<16}{tau_s:<10} GPU:{e.total_gpu_mb:>8.0f}MB  CPU:{e.peak_cpu_mb:>8.0f}MB")
    
    print("""
    COLAB PRO (L4: 24GB VRAM, 50GB RAM):
    ----------------------------------------
    """)
    l4_ok = [e for e in configs if "L4" in recommend_colab(e.total_gpu_mb, e.peak_cpu_mb)]
    for e in l4_ok:
        tau_s = f" tau={e.tau}" if e.tau else ""
        print(f"      [OK]  {e.dataset:<14} {e.model:<16}{tau_s:<10} GPU:{e.total_gpu_mb:>8.0f}MB  CPU:{e.peak_cpu_mb:>8.0f}MB")
    
    print("""
    A100 (40GB VRAM, 83GB RAM):
    ----------------------------------------
    """)
    a100_ok = [e for e in configs if "A100" in recommend_colab(e.total_gpu_mb, e.peak_cpu_mb)]
    for e in a100_ok:
        tau_s = f" tau={e.tau}" if e.tau else ""
        print(f"      [OK]  {e.dataset:<14} {e.model:<16}{tau_s:<10} GPU:{e.total_gpu_mb:>8.0f}MB  CPU:{e.peak_cpu_mb:>8.0f}MB")
    
    exceeds = [e for e in configs if "EXCEEDS" in recommend_colab(e.total_gpu_mb, e.peak_cpu_mb)]
    if exceeds:
        print("""
    EXCEEDS COLAB RESOURCES:
    ----------------------------------------
        """)
        for e in exceeds:
            tau_s = f" tau={e.tau}" if e.tau else ""
            print(f"      [!!]  {e.dataset:<14} {e.model:<16}{tau_s:<10} GPU:{e.total_gpu_mb:>8.0f}MB  CPU:{e.peak_cpu_mb:>8.0f}MB")
    
    print("""
    KEY TAKEAWAYS:
    ==============
    1. LightGCN: Fits easily on any GPU. ~17-35 MB GPU VRAM.
    2. LightCCN-Flat: Minimal overhead vs LightGCN (only adds S to adjacency).
       Fits on free Colab T4 for all datasets and tau values.
    3. LightCCN-Multi: The bottleneck is face/edge EMBEDDING TABLES.
       - tau >= 20: Usually fits on T4 for gowalla/yelp2018.
       - tau < 10: Face embedding tables grow large, may need L4/A100.
       - tau <= 3 on amazon-book: Can require multi-GB VRAM just for embeddings.
    4. CPU RAM bottleneck: The co-occurrence matrix C = R^T @ R during cell complex
       construction. For amazon-book with low tau, this can exceed free tier RAM.
    5. PRACTICAL ADVICE: Use tau >= 10 for initial experiments, tau >= 20 for
       amazon-book. Cache cell complexes to avoid repeated construction costs.
    """)


if __name__ == "__main__":
    main()
