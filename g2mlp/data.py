"""Dataset loading, splits and split fingerprints.

Derived from GLNN (Copyright Snap Inc. 2021, MIT License, see LICENSE).
The CPF loader is adapted from the CPF implementation
https://github.com/BUPT-GAMMA/CPF/tree/389c01aaf238689ee7b1e5aba127842341e123b6/data
eliminate_self_loops_adj, largest_connected_component and binarize_labels trace
back to gnn-benchmark (https://github.com/shchur/gnn-benchmark, MIT License, see
LICENSE).
"""
import hashlib
from pathlib import Path

import dgl
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import LabelBinarizer

from .utils import REPO_ROOT

CPF_DATASETS = ["cora", "citeseer", "pubmed", "a-computer", "a-photo"]
OGB_DATASETS = ["ogbn-arxiv"]
DEFAULT_DATA_PATH = REPO_ROOT / "data"


def load_data(dataset, data_path=None, seed=0, labelrate_train=20, labelrate_val=30):
    """Return (g, labels, idx_train, idx_val, idx_test).

    g is symmetric and carries one self-loop per node, so g.edges() gives the
    closed neighborhoods N+(v) of the paper.
    """
    data_path = Path(data_path) if data_path is not None else DEFAULT_DATA_PATH
    if dataset in CPF_DATASETS:
        return load_cpf_data(dataset, data_path, seed, labelrate_train, labelrate_val)
    if dataset in OGB_DATASETS:
        return load_ogb_data(dataset, data_path)
    raise ValueError(f"Unknown dataset: {dataset}")


def load_ogb_data(dataset, data_path):
    from ogb.nodeproppred import DglNodePropPredDataset

    data = DglNodePropPredDataset(dataset, str(data_path))
    split = data.get_idx_split()
    idx_train, idx_val, idx_test = split["train"], split["valid"], split["test"]

    g, labels = data[0]
    labels = labels.squeeze()

    if dataset == "ogbn-arxiv":
        # Add reversed edges so the graph is symmetric.
        src, dst = g.all_edges()
        g.add_edges(dst, src)
        g = g.remove_self_loop().add_self_loop()
    return g, labels, idx_train, idx_val, idx_test


def load_cpf_data(dataset, data_path, seed, labelrate_train, labelrate_val):
    path = Path(data_path) / f"{dataset}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. See the Data section of README.md.")
    data = load_npz_to_sparse_graph(path)

    # Remove self-loops and keep the largest connected component.
    data = data.standardize()
    adj, features, labels = data.unpack()
    labels = binarize_labels(labels)

    random_state = np.random.RandomState(seed)
    idx_train, idx_val, idx_test = get_train_val_test_split(
        random_state, labels, labelrate_train, labelrate_val
    )

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels.argmax(axis=1))

    adj = normalize_adj(adj).tocoo()
    g = dgl.graph((adj.row, adj.col), num_nodes=adj.shape[0])
    g.ndata["feat"] = features
    return (g, labels, torch.LongTensor(idx_train), torch.LongTensor(idx_val),
            torch.LongTensor(idx_test))


def idx_split(idx, ratio, seed=0):
    """Randomly split idx into a ratio portion and a (1 - ratio) portion."""
    n = len(idx)
    cut = int(n * ratio)
    generator = torch.Generator()
    generator.manual_seed(seed)
    shuffle = torch.randperm(n, generator=generator)
    return idx[shuffle[:cut]], idx[shuffle[cut:]]


def graph_split(idx_train, idx_val, idx_test, rate, seed):
    """Hold out a `rate` share of the test nodes for the inductive evaluation.

    Indices starting with obs_ refer to positions inside the observed graph;
    idx_obs and idx_test_ind refer to the original graph.
    """
    idx_test_ind, idx_test_tran = idx_split(idx_test, rate, seed)
    idx_obs = torch.cat([idx_train, idx_val, idx_test_tran])
    n_train, n_val = idx_train.shape[0], idx_val.shape[0]
    positions = torch.arange(idx_obs.shape[0])
    obs_idx_train = positions[:n_train]
    obs_idx_val = positions[n_train:n_train + n_val]
    obs_idx_test = positions[n_train + n_val:]
    return obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind


def split_fingerprint(*index_sets):
    """sha1 of the sorted index sets: split_fingerprint(idx_train, idx_val, idx_test).

    Teachers store this value and students compare it. Further index sets, such
    as the inductive test nodes, may follow the first three.
    """
    h = hashlib.sha1()
    for idx in index_sets:
        array = idx.cpu().numpy() if torch.is_tensor(idx) else np.asarray(idx)
        array = np.sort(array.astype(np.int64))
        h.update(np.int64(array.size).tobytes())
        h.update(array.tobytes())
    return h.hexdigest()


class SparseGraph:
    """Attributed labeled graph stored in sparse matrix form."""

    def __init__(self, adj_matrix, attr_matrix=None, labels=None):
        if not sp.isspmatrix(adj_matrix):
            raise ValueError(f"Adjacency matrix must be sparse, got {type(adj_matrix)}")
        adj_matrix = adj_matrix.tocsr().astype(np.float32)
        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree")
        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError(f"Unsupported attribute matrix: {type(attr_matrix)}")
            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Adjacency and attribute matrices don't agree")
        if labels is not None and labels.shape[0] != adj_matrix.shape[0]:
            raise ValueError("Adjacency matrix and label vector don't agree")
        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels

    def standardize(self):
        """Select the LCC of the unweighted, undirected, no-self-loop graph."""
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        self.adj_matrix = self.adj_matrix + self.adj_matrix.T
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        self.adj_matrix = eliminate_self_loops_adj(self.adj_matrix)
        return largest_connected_component(self)

    def unpack(self):
        return self.adj_matrix, self.attr_matrix, self.labels


def load_npz_to_sparse_graph(file_name):
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
    adj_matrix = sp.csr_matrix(
        (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
        shape=loader["adj_shape"],
    )
    if "attr_data" in loader:
        attr_matrix = sp.csr_matrix(
            (loader["attr_data"], loader["attr_indices"], loader["attr_indptr"]),
            shape=loader["attr_shape"],
        )
    else:
        attr_matrix = loader.get("attr_matrix")
    if "labels_data" in loader:
        labels = sp.csr_matrix(
            (loader["labels_data"], loader["labels_indices"], loader["labels_indptr"]),
            shape=loader["labels_shape"],
        )
    else:
        labels = loader.get("labels")
    return SparseGraph(adj_matrix, attr_matrix, labels)


def normalize_adj(adj):
    """Row-normalized A + I."""
    adj = adj + sp.eye(adj.shape[0])
    inverse = np.power(np.array(adj.sum(1)), -1.0).flatten()
    inverse[np.isinf(inverse)] = 0.0
    return sp.diags(inverse).dot(adj)


def eliminate_self_loops_adj(adj):
    adj = adj.tolil()
    adj.setdiag(0)
    adj = adj.tocsr()
    adj.eliminate_zeros()
    return adj


def largest_connected_component(graph):
    _, components = connected_components(graph.adj_matrix)
    largest = np.argmax(np.bincount(components))
    keep = np.flatnonzero(components == largest)
    graph.adj_matrix = graph.adj_matrix[keep][:, keep]
    if graph.attr_matrix is not None:
        graph.attr_matrix = graph.attr_matrix[keep]
    if graph.labels is not None:
        graph.labels = graph.labels[keep]
    return graph


def binarize_labels(labels):
    return LabelBinarizer().fit_transform(labels).astype(np.float32)


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    """Draw a fixed number of examples per class."""
    num_classes = labels.shape[1]
    samples = []
    for class_index in range(num_classes):
        candidates = np.flatnonzero(labels[:, class_index] > 0.0)
        if forbidden_indices is not None:
            candidates = np.setdiff1d(candidates, forbidden_indices)
        samples.append(random_state.choice(candidates, num_examples_per_class, replace=False))
    return np.concatenate(samples)


def get_train_val_test_split(random_state, labels, train_examples_per_class,
                             val_examples_per_class):
    """Fixed label rate per class for train and val; the rest is test."""
    train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    val_indices = sample_per_class(random_state, labels, val_examples_per_class,
                                   forbidden_indices=train_indices)
    forbidden = np.concatenate((train_indices, val_indices))
    test_indices = np.setdiff1d(np.arange(labels.shape[0]), forbidden)
    return train_indices, val_indices, test_indices
