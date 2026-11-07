"""Node-classification student, transductive and inductive.

The student trains on the observed nodes: all nodes in the transductive
setting, and all nodes except the inductive test nodes in the inductive
setting. Inductive test nodes enter only through their features at inference.
"""
import copy
from dataclasses import dataclass

import dgl
import torch
import torch.nn.functional as F
import torch.optim as optim

from . import losses
from .artifacts import load_teacher, teacher_dir
from .curvature import graph_name, node_curvature
from .data import graph_split, load_data
from .models import CAWTransform, Model
from .teacher import SPLIT_RATE, nc_fingerprint
from .utils import accuracy, model_conf, set_seed


@dataclass
class NCContext:
    """Data of one (dataset, teacher, setting, seed) cell.

    Tensors hold observed nodes and idx_* are positions among them; feats_ind
    and labels_ind hold the inductive test nodes.
    """
    dataset: str
    teacher: str
    setting: str
    seed: int
    device: torch.device
    graph: dgl.DGLGraph
    op: losses.NeighborhoodOp
    feats: torch.Tensor
    labels: torch.Tensor
    idx_train: torch.Tensor
    idx_val: torch.Tensor
    idx_test: torch.Tensor
    feats_ind: torch.Tensor
    labels_ind: torch.Tensor
    out_t: torch.Tensor
    logits_t: torch.Tensor
    h_t: torch.Tensor
    num_classes: int
    fingerprint: str
    teacher_id: str

    def kappa(self, cfg):
        """Node curvature of the observed graph."""
        return node_curvature(self.graph, cfg["curv_kind"], self.dataset,
                              graph_name(self.setting, self.seed),
                              alpha=cfg["curv_alpha"]).to(self.device)


def observed_graph(g, idx_obs):
    """Subgraph on idx_obs, node i being idx_obs[i], with one self-loop per node."""
    obs_g = g.subgraph(idx_obs)
    if not torch.equal(obs_g.ndata[dgl.NID].cpu(), idx_obs.cpu()):
        raise RuntimeError("Observed graph does not follow the order of idx_obs")
    return dgl.add_self_loop(dgl.remove_self_loop(obs_g))


def build_context(dataset, teacher, setting, seed, device, data_path=None,
                  teacher_root=None, split_rate=SPLIT_RATE):
    """Load data and verify the teacher for one seed."""
    g, labels, idx_train, idx_val, idx_test = load_data(dataset, data_path, seed=seed)
    feats = g.ndata["feat"]
    num_classes = int(labels.max()) + 1
    fingerprint = nc_fingerprint(setting, idx_train, idx_val, idx_test, split_rate, seed)
    directory = teacher_dir(setting, dataset, teacher, seed, teacher_root)
    expected = {"dataset": dataset, "task": "nc", "setting": setting, "seed": seed,
                "teacher": teacher}
    config = model_conf(teacher, dataset)

    if setting == "tran":
        obs_g = dgl.add_self_loop(dgl.remove_self_loop(g))
        idx_obs = torch.arange(g.num_nodes())
        idx_test_ind = torch.empty(0, dtype=torch.long)
        obs_idx_train, obs_idx_val, obs_idx_test = idx_train, idx_val, idx_test
        artifacts = load_teacher(directory, fingerprint, expected, config, labels, idx_val,
                                 num_classes)
    elif setting == "ind":
        obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = graph_split(
            idx_train, idx_val, idx_test, split_rate, seed)
        obs_g = observed_graph(g, idx_obs)
        artifacts = load_teacher(directory, fingerprint, expected, config, labels,
                                 idx_obs[obs_idx_val], num_classes, idx_read=idx_obs,
                                 idx_masked=idx_test_ind)
    else:
        raise ValueError(f"Unknown setting: {setting}")

    src, dst = obs_g.edges()
    return NCContext(
        dataset=dataset, teacher=teacher, setting=setting, seed=seed, device=device,
        graph=obs_g, op=losses.NeighborhoodOp(src.to(device), dst.to(device),
                                              obs_g.num_nodes()),
        feats=feats[idx_obs].to(device), labels=labels[idx_obs].to(device),
        idx_train=obs_idx_train, idx_val=obs_idx_val, idx_test=obs_idx_test,
        feats_ind=feats[idx_test_ind].to(device), labels_ind=labels[idx_test_ind].to(device),
        out_t=artifacts.out[idx_obs].to(device), logits_t=artifacts.logits[idx_obs].to(device),
        h_t=artifacts.h[idx_obs].to(device), num_classes=num_classes,
        fingerprint=fingerprint, teacher_id=artifacts.identity)


def student_conf(feat_dim, label_dim, hidden_dim, cfg):
    """MLP settings of configs/models.yaml with the teacher's penultimate size."""
    return dict(cfg["student_model"], feat_dim=feat_dim, label_dim=label_dim,
                hidden_dim=hidden_dim)


def objective(ctx, cfg, logits, h_s, log_t, alpha, beta, transform):
    """Eq. (13) of the paper, with L_CAW of Eq. (12).

    The CE term is added only when lambda_CE > 0. With caw_transform `student`
    or `both` a head maps h_S, and `both` also maps h_T, before the feature W2
    term.
    """
    tau = cfg["tau"]
    log_s = losses.soften(logits, tau)
    loss = cfg["lambda_kd"] * losses.kd_loss(log_s, log_t, tau)
    if cfg["lambda_ce"] > 0:
        loss = loss + cfg["lambda_ce"] * F.cross_entropy(logits[ctx.idx_train],
                                                         ctx.labels[ctx.idx_train])
    h_s_mapped, h_t_mapped = transform(h_s, ctx.h_t)
    l_h = losses.h_term(h_s_mapped, h_t_mapped, ctx.op, beta, cfg["w2_eps"])
    l_z = losses.z_term(logits, ctx.logits_t, ctx.op, alpha, cfg["w2_eps"])
    return loss + losses.caw_penalty(l_h, l_z, cfg["lambda_caw"], cfg["eps_sg"])


def train_student(ctx, cfg):
    """Train one student. Early stopping on val accuracy with `>=`."""
    set_seed(ctx.seed)
    conf = student_conf(ctx.feats.shape[1], ctx.num_classes, ctx.h_t.shape[1], cfg)
    student = Model(dict(conf, device=ctx.device))
    transform = CAWTransform(cfg["caw_transform"], conf["hidden_dim"]).to(ctx.device)
    params = list(student.parameters())
    params += [p for p in transform.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=conf["learning_rate"], weight_decay=conf["weight_decay"])

    alpha, beta = losses.node_weights(ctx.kappa(cfg), cfg["gamma_z"], cfg["gamma_h"])
    log_t = losses.soften(ctx.out_t, cfg["tau"])

    best_val, best_state, count = -1.0, None, 0
    for epoch in range(1, cfg["max_epoch"] + 1):
        student.train()
        transform.train()
        optimizer.zero_grad()
        h_list, logits = student.encode(None, ctx.feats)
        loss = objective(ctx, cfg, logits, h_list[-1], log_t, alpha, beta, transform)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {ctx.dataset} "
                                     f"seed {ctx.seed}")
        loss.backward()
        optimizer.step()

        student.eval()
        with torch.no_grad():
            score = accuracy(student(None, ctx.feats[ctx.idx_val]), ctx.labels[ctx.idx_val])
        if score >= best_val:
            best_val, count = score, 0
            best_state = copy.deepcopy(student.state_dict())
        else:
            count += 1
        if count >= cfg["patience"]:
            break
    return {"val": best_val, "state": best_state, "conf": conf}


def predict(student, feats):
    student.eval()
    with torch.no_grad():
        return student(None, feats)


def test_scores(student, ctx):
    """Test metrics of the student, from node features alone."""
    logits = predict(student, ctx.feats[ctx.idx_test])
    observed = accuracy(logits, ctx.labels[ctx.idx_test])
    if ctx.setting == "tran":
        return {"test": observed}
    inductive = accuracy(predict(student, ctx.feats_ind), ctx.labels_ind)
    return {"test_tran": observed, "test_ind": inductive,
            "test_prod": 0.8 * observed + 0.2 * inductive}


def run_seed(ctx, cfg):
    """Train the student of one seed; return its scores and checkpoint."""
    best = train_student(ctx, cfg)
    student = Model(dict(best["conf"], device=ctx.device))
    student.load_state_dict(best["state"])
    scores = test_scores(student, ctx)
    checkpoint = {
        "task": "nc", "dataset": ctx.dataset, "teacher": ctx.teacher, "setting": ctx.setting,
        "seed": ctx.seed, "state": best["state"],
        "model_conf": best["conf"], "config": cfg, "split_fingerprint": ctx.fingerprint,
    }
    return dict(scores, val=best["val"]), checkpoint


def evaluate_checkpoint(checkpoint, device, data_path=None, split_rate=SPLIT_RATE):
    """Recompute the metrics of a checkpoint from node features."""
    dataset, seed, setting = (checkpoint[key] for key in ("dataset", "seed", "setting"))
    g, labels, idx_train, idx_val, idx_test = load_data(dataset, data_path, seed=seed)
    fingerprint = nc_fingerprint(setting, idx_train, idx_val, idx_test, split_rate, seed)
    if fingerprint != checkpoint["split_fingerprint"]:
        raise RuntimeError("Checkpoint was trained on another split")
    feats, labels = g.ndata["feat"].to(device), labels.to(device)
    student = Model(dict(checkpoint["model_conf"], device=device))
    student.load_state_dict(checkpoint["state"])
    logits = predict(student, feats)
    if setting == "tran":
        return {"val": accuracy(logits[idx_val], labels[idx_val]),
                "test": accuracy(logits[idx_test], labels[idx_test])}
    obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = graph_split(
        idx_train, idx_val, idx_test, split_rate, seed)[1:]
    observed = accuracy(logits[idx_obs[obs_idx_test]], labels[idx_obs[obs_idx_test]])
    inductive = accuracy(logits[idx_test_ind], labels[idx_test_ind])
    return {"val": accuracy(logits[idx_obs[obs_idx_val]], labels[idx_obs[obs_idx_val]]),
            "test_tran": observed, "test_ind": inductive,
            "test_prod": 0.8 * observed + 0.2 * inductive}
