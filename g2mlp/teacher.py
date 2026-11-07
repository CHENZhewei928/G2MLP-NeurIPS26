"""Node-classification teacher training.

Derived from GLNN (Copyright Snap Inc. 2021, MIT License, see LICENSE).
"""
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import DataLoader, NeighborSampler

from .artifacts import save_teacher, teacher_dir
from .data import graph_split, load_data, split_fingerprint
from .models import Model
from .utils import accuracy, model_conf, set_seed

SPLIT_RATE = 0.2


def sage_step(model, dataloader, feats, labels, optimizer):
    """One epoch of neighbor-sampled mini-batches."""
    device = feats.device
    model.train()
    for input_nodes, output_nodes, blocks in dataloader:
        blocks = [block.int().to(device) for block in blocks]
        out = model(blocks, feats[input_nodes]).log_softmax(dim=1)
        loss = F.nll_loss(out, labels[output_nodes])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def full_batch_step(model, g, feats, labels, idx_train, optimizer):
    model.train()
    out = model(g, feats).log_softmax(dim=1)
    loss = F.nll_loss(out[idx_train], labels[idx_train])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def make_step(conf, model, g, feats, labels, idx_train, optimizer):
    """Return a closure running one training epoch on `g`."""
    device = conf["device"]
    if "SAGE" in model.model_name:
        g.create_formats_()
        sampler = NeighborSampler([int(f) for f in conf["fan_out"].split(",")])
        dataloader = DataLoader(g, idx_train, sampler, batch_size=conf["batch_size"],
                                shuffle=True, drop_last=False,
                                num_workers=conf["num_workers"])
        return lambda: sage_step(model, dataloader, feats, labels, optimizer)
    g_device = g.to(device)
    return lambda: full_batch_step(model, g_device, feats, labels, idx_train, optimizer)


def full_forward(model, g, feats):
    model.eval()
    with torch.no_grad():
        return model.encode_full(g, feats)


def fit(model, step, val_score, max_epoch, patience):
    """Early stopping on validation accuracy with `>=`."""
    best_val, count, state = 0.0, 0, None
    for _ in range(max_epoch):
        step()
        score = val_score()
        if score >= best_val:
            best_val, count = score, 0
            state = copy.deepcopy(model.state_dict())
        else:
            count += 1
        if count == patience:
            break
    model.load_state_dict(state)


def run_transductive(conf, model, g, feats, labels, indices, optimizer):
    set_seed(conf["seed"])
    device = conf["device"]
    idx_train, idx_val, idx_test = indices
    feats, labels = feats.to(device), labels.to(device)
    step = make_step(conf, model, g, feats, labels, idx_train, optimizer)
    g_device = g.to(device)

    def val_score():
        logits = full_forward(model, g_device, feats)[1]
        return accuracy(logits[idx_val], labels[idx_val])

    fit(model, step, val_score, conf["max_epoch"], conf["patience"])
    h_list, logits = full_forward(model, g_device, feats)
    return {
        "logits": logits,
        "h": h_list[-1],
        "val": accuracy(logits[idx_val], labels[idx_val]),
        "test_scores": {"test": accuracy(logits[idx_test], labels[idx_test])},
    }


def run_inductive(conf, model, g, feats, labels, indices, optimizer):
    """Train on the observed graph; inductive test nodes appear only at test time.

    Saved outputs are computed on the observed graph and are NaN at inductive
    test nodes. The inductive test score is computed on the full graph, whose
    outputs are not saved.
    """
    set_seed(conf["seed"])
    device = conf["device"]
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices
    feats, labels = feats.to(device), labels.to(device)
    obs_feats, obs_labels = feats[idx_obs], labels[idx_obs]
    obs_g = g.subgraph(idx_obs)
    step = make_step(conf, model, obs_g, obs_feats, obs_labels, obs_idx_train, optimizer)
    obs_g_device = obs_g.to(device)

    def val_score():
        logits = full_forward(model, obs_g_device, obs_feats)[1]
        return accuracy(logits[obs_idx_val], obs_labels[obs_idx_val])

    fit(model, step, val_score, conf["max_epoch"], conf["patience"])
    h_list, obs_logits = full_forward(model, obs_g_device, obs_feats)
    full_logits = full_forward(model, g.to(device), feats)[1]

    test_tran = accuracy(obs_logits[obs_idx_test], obs_labels[obs_idx_test])
    test_ind = accuracy(full_logits[idx_test_ind], labels[idx_test_ind])
    num_nodes = g.num_nodes()
    logits = torch.full((num_nodes, obs_logits.shape[1]), float("nan"), device=device)
    logits[idx_obs] = obs_logits
    h = torch.full((num_nodes, h_list[-1].shape[1]), float("nan"), device=device)
    h[idx_obs] = h_list[-1]
    return {
        "logits": logits,
        "h": h,
        "val": accuracy(obs_logits[obs_idx_val], obs_labels[obs_idx_val]),
        "test_scores": {"test_tran": test_tran, "test_ind": test_ind,
                        "test_prod": 0.8 * test_tran + 0.2 * test_ind},
    }


def nc_fingerprint(setting, idx_train, idx_val, idx_test, rate=SPLIT_RATE, seed=0):
    """Fingerprint of the split a node-classification run uses."""
    if setting == "tran":
        return split_fingerprint(idx_train, idx_val, idx_test)
    _, _, _, idx_obs, idx_test_ind = graph_split(idx_train, idx_val, idx_test, rate, seed)
    return split_fingerprint(idx_train, idx_val, idx_obs, idx_test_ind)


def teacher_conf(teacher, dataset, max_epoch=None, patience=None):
    """Settings of a teacher from configs/models.yaml, with optional schedule overrides."""
    conf = model_conf(teacher, dataset)
    if max_epoch is not None:
        conf["max_epoch"] = max_epoch
    if patience is not None:
        conf["patience"] = patience
    return conf


def train_nc_teacher(dataset, teacher, setting, seed, device, data_path=None, root=None,
                     max_epoch=None, patience=None, split_rate=SPLIT_RATE):
    """Train one teacher, write its artifacts and return its meta dictionary.

    max_epoch and patience default to the values in configs/models.yaml.
    """
    set_seed(seed)
    g, labels, idx_train, idx_val, idx_test = load_data(dataset, data_path, seed=seed)
    feats = g.ndata["feat"]
    conf = teacher_conf(teacher, dataset, max_epoch, patience)
    conf.update(feat_dim=feats.shape[1], label_dim=int(labels.max()) + 1, device=device,
                seed=seed)

    model = Model(conf)
    optimizer = optim.Adam(model.parameters(), lr=conf["learning_rate"],
                           weight_decay=conf["weight_decay"])
    if setting == "tran":
        result = run_transductive(conf, model, g, feats, labels,
                                  (idx_train, idx_val, idx_test), optimizer)
    elif setting == "ind":
        indices = graph_split(idx_train, idx_val, idx_test, split_rate, seed)
        result = run_inductive(conf, model, g, feats, labels, indices, optimizer)
    else:
        raise ValueError(f"Unknown setting: {setting}")

    saved_conf = {k: v for k, v in conf.items() if k != "device"}
    meta = {
        "dataset": dataset, "task": "nc", "setting": setting, "seed": seed,
        "teacher": teacher, "num_nodes": g.num_nodes(),
        "split_fingerprint": nc_fingerprint(setting, idx_train, idx_val, idx_test,
                                            split_rate, seed),
        "val_score": result["val"], "test_scores": result["test_scores"],
        "config": saved_conf,
    }
    save_teacher(teacher_dir(setting, dataset, teacher, seed, root), meta,
                 out=result["logits"].log_softmax(dim=1), h_teacher=result["h"],
                 model_state=model.state_dict(), logits=result["logits"])
    return meta
