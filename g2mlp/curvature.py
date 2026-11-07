"""Node curvature with a cache.

A cache file is named by dataset, kind and graph, for example cora_orc.npz for
the full graph and cora_orc_ind_seed3.npz for the observed graph of an
inductive seed. It stores the node vector and a hash of the graph, and is
recomputed when the graph differs.

Kinds:
    orc         Ollivier-Ricci via GraphRicciCurvature, W1 between lazy random walks
    approx_orc  closed-form approximation of ORC from common-neighbor counts

The node value is the mean of the incident edge curvatures, the kappa_bar_v of
the paper. Isolated nodes get 0.
"""
import hashlib
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .utils import REPO_ROOT

KINDS = ("orc", "approx_orc")


def undirected_edges(g):
    """Return (n, edges) with edges [m, 2], u < v, sorted and unique.

    Self-loops are dropped and direction is ignored.
    """
    n = g.num_nodes()
    src, dst = g.edges()
    src = src.cpu().numpy().astype(np.int64)
    dst = dst.cpu().numpy().astype(np.int64)
    keep = src != dst
    lo = np.minimum(src[keep], dst[keep])
    hi = np.maximum(src[keep], dst[keep])
    keys = np.unique(lo * n + hi)
    return n, np.stack([keys // n, keys % n], axis=1)


def graph_hash(n, edges):
    """sha1 over n and the sorted undirected edge list, truncated to 16 hex."""
    h = hashlib.sha1()
    h.update(np.int64(n).tobytes())
    h.update(np.asarray(edges, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def graph_name(setting, seed):
    """Name of the graph a student reads: the full graph, or an observed graph."""
    return "" if setting == "tran" else f"ind_seed{seed}"


def cache_path(dataset, kind, name="", alpha=0.5, cache_dir="curvature"):
    root = Path(cache_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    tag = kind
    if alpha != 0.5:
        tag = f"{kind}{alpha:g}"
    stem = f"{dataset}_{tag}_{name}" if name else f"{dataset}_{tag}"
    return root / f"{stem}.npz"


def _adjacency(n, edges):
    import scipy.sparse as sp

    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(rows), dtype=np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def _networkx_graph(n, edges):
    import networkx as nx

    nx_g = nx.Graph()
    nx_g.add_nodes_from(range(n))
    nx_g.add_edges_from(edges.tolist())
    return nx_g


def _patch_orc_for_windows(orc_module):
    """Run GraphRicciCurvature serially on Windows.

    The library uses a multiprocessing Pool, which deadlocks under Windows
    spawn semantics. Adapted from OllivierRicci._compute_ricci_curvature_edges
    of GraphRicciCurvature (Apache License 2.0).
    """
    if getattr(orc_module, "_g2mlp_serial", False):
        return

    import networkit as nk

    def compute_edges(G, weight="weight", edge_list=(), alpha=0.5, **kwargs):
        orc_module._get_single_node_neighbors_distributions.cache_clear()
        orc_module._source_target_shortest_path.cache_clear()
        orc_module._Gk = nk.nxadapter.nx2nk(G, weightAttr=weight)
        orc_module._alpha = alpha
        orc_module._weight = weight
        for name in ("method", "base", "exp_power", "nbr_topk", "shortest_path"):
            if name in kwargs:
                setattr(orc_module, f"_{name}", kwargs[name])

        to_nk = {node: idx for idx, node in enumerate(G.nodes())}
        to_nx = {idx: node for node, idx in to_nk.items()}
        if orc_module._shortest_path == "all_pairs":
            orc_module._apsp = orc_module._get_all_pairs_shortest_path()

        pairs = [(to_nk[s], to_nk[t]) for s, t in (edge_list if edge_list else G.edges())]
        result = [orc_module._wrap_compute_single_edge(pair) for pair in pairs]
        return {(to_nx[k[0]], to_nx[k[1]]): rc[k] for rc in result for k in rc}

    orc_module._compute_ricci_curvature_edges = compute_edges
    orc_module._g2mlp_serial = True


def available_cpus():
    """CPUs available to this process."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _orc_edges(n, edges, alpha=0.5):
    import GraphRicciCurvature.OllivierRicci as orc_module

    if platform.system() == "Windows":
        _patch_orc_for_windows(orc_module)
    orc = orc_module.OllivierRicci(_networkx_graph(n, edges), alpha=alpha,
                                   proc=available_cpus())
    orc.compute_ricci_curvature()
    return np.array([orc.G[u][v]["ricciCurvature"] for u, v in edges.tolist()])


def _approx_orc_edges(n, edges, alpha=0.5):
    """kappa(i, j) = 1 - W1 with
    W1 = alpha + (1 - alpha) * (1 - |N(i) & N(j)| / sqrt(d_i d_j)).
    """
    adj = _adjacency(n, edges)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    common = np.asarray((adj @ adj)[edges[:, 0], edges[:, 1]]).ravel()
    scale = np.sqrt(np.maximum(deg[edges[:, 0]] * deg[edges[:, 1]], 1.0))
    w1 = alpha + (1.0 - alpha) * (1.0 - common / scale)
    return 1.0 - w1


def compute_node_curvature(n, edges, kind, alpha=0.5):
    """Mean incident edge curvature per node, as a float32 array of length n."""
    if kind == "orc":
        edge_curv = _orc_edges(n, edges, alpha)
    elif kind == "approx_orc":
        edge_curv = _approx_orc_edges(n, edges, alpha)
    else:
        raise ValueError(f"Unknown curvature kind {kind}; expected one of {KINDS}")
    total = np.zeros(n, dtype=np.float64)
    deg = np.zeros(n, dtype=np.float64)
    for column in (0, 1):
        np.add.at(total, edges[:, column], edge_curv)
        np.add.at(deg, edges[:, column], 1.0)
    return (total / np.maximum(deg, 1.0)).astype(np.float32)


def node_curvature(g, kind, dataset, name="", alpha=0.5, cache_dir="curvature"):
    """Node curvature of `g` as a float32 tensor on CPU.

    `name` tells graphs of one dataset apart, see graph_name.
    """
    n, edges = undirected_edges(g)
    ghash = graph_hash(n, edges)
    path = cache_path(dataset, kind, name, alpha, cache_dir)
    kappa = None
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if str(stored["graph"]) == ghash and stored["kappa"].shape == (n,):
                kappa = stored["kappa"]
    if kappa is None:
        start = time.time()
        kappa = compute_node_curvature(n, edges, kind, alpha)
        print(f"[curvature] {kind} on {dataset} ({n} nodes, {len(edges)} edges) "
              f"in {time.time() - start:.1f}s", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, kappa=kappa, graph=np.array(ghash))
        os.replace(tmp, path)
    return torch.from_numpy(kappa.astype(np.float32))
