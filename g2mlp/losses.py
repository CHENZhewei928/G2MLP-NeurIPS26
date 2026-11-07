"""The G2MLP objective, Eqs. (10)-(13) of the paper.

Eq. (10): W2diag, the closed-form squared W2 distance between diagonal
Gaussians fitted to the teacher and student neighborhood measures,
||m_T - m_S||^2 + ||sqrt(sigma_T + eps) - sqrt(sigma_S + eps)||^2.
Eq. (11): mean-preserving curvature weights alpha_v = N softmax(-gamma_z k)_v
and beta_v = N softmax(+gamma_h k)_v, with k the z-scored node curvature.
Eq. (12): L_CAW = mean_v [beta_v W2diag(h_T, h_S) + alpha_v W2diag(z_T, z_S)],
with h the penultimate features and z the raw logits.
Eq. (13): L = lambda_CE CE + lambda_KD KD + lambda_CAW L_CAW / (sg(L_CAW) + eps_sg).

Teacher quantities are constants; only the student side carries gradients.
"""
import torch
import torch.nn.functional as F

EPS_SG = 1e-8


class NeighborhoodOp:
    """Row-normalized neighborhood operator Q.

    Q[v, u] is the number of edges (v, u) divided by the number of edges
    leaving v. With one self-loop per node and a symmetric edge list, the
    nonzeros of row v are the closed neighborhood N+(v) of the paper.

    Q is stored with a leading batch dimension so that torch.bmm can apply it.
    """

    def __init__(self, src, dst, n, dtype=torch.float32):
        src = torch.as_tensor(src, dtype=torch.long)
        dst = torch.as_tensor(dst, dtype=torch.long, device=src.device)
        self.n = int(n)
        self.deg = torch.zeros(self.n, dtype=dtype, device=src.device)
        self.deg.index_add_(0, src, torch.ones_like(src, dtype=dtype))
        indices = torch.stack([torch.zeros_like(src), src, dst])
        values = 1.0 / self.deg[src]
        self.q = torch.sparse_coo_tensor(indices, values, (1, self.n, self.n)).coalesce()

    @property
    def active(self):
        """Mask of nodes with a non-empty neighborhood."""
        return self.deg > 0

    def mean(self, x):
        """Neighborhood mean Q @ x, the m_v of Eq. (10)."""
        if x.dim() == 1:
            return self.mean(x.unsqueeze(1)).squeeze(1)
        return torch.bmm(self.q, x.unsqueeze(0)).squeeze(0)

    def moments(self, x):
        """Return (mean, variance) over each neighborhood."""
        m = self.mean(x)
        return m, (self.mean(x * x) - m**2).clamp_min(0)


def curvature_weights(kappa_bar, gamma):
    """Mean-preserving curvature weights, Eq. (11).

    w_v = N * softmax(gamma * k)_v with k = (kappa_bar - mean) / std, which is
    exp(gamma * k_v) divided by its mean over nodes, computed without overflow.
    Constant curvature gives all ones.
    """
    std = kappa_bar.std(unbiased=False)
    k = (kappa_bar - kappa_bar.mean()) / std if std > 0 else torch.zeros_like(kappa_bar)
    return k.numel() * torch.softmax(gamma * k, dim=0)


def node_weights(kappa_bar, gamma_z, gamma_h):
    """Return (alpha_v, beta_v) of Eq. (11).

    alpha_v upweights low-curvature nodes, beta_v high-curvature nodes.
    """
    alpha = curvature_weights(kappa_bar, -gamma_z)
    beta = curvature_weights(kappa_bar, gamma_h)
    return alpha, beta


def w2_diag(student, teacher, op, eps=1e-6):
    """Per-node closed-form squared W2 between diagonal Gaussians, Eq. (10)."""
    m_s, v_s = op.moments(student)
    m_t, v_t = op.moments(teacher)
    mean_term = (m_t - m_s).pow(2).sum(dim=1)
    cov_term = ((v_t + eps).sqrt() - (v_s + eps).sqrt()).pow(2).sum(dim=1)
    return mean_term + cov_term


def soften(log_p, tau):
    """Log-probabilities at temperature tau, from logits or log-probabilities."""
    return (log_p / tau).log_softmax(dim=1)


def kd_loss(log_s, log_t, tau=1.0):
    """tau^2 * KL(t || s), averaged over the nodes given.

    Both inputs are log-probabilities from `soften` with the same tau.
    """
    return tau**2 * F.kl_div(log_s, log_t, reduction="batchmean", log_target=True)


def h_term(h_s, h_t, op, beta, eps=1e-6):
    """Feature block of Eq. (12): mean_v beta_v W2diag(h_T, h_S)."""
    mask = op.active
    return (beta[mask] * w2_diag(h_s, h_t, op, eps)[mask]).mean()


def z_term(logits_s, logits_t, op, alpha, eps=1e-6):
    """Logit block of Eq. (12): mean_v alpha_v W2diag(z_T, z_S) on raw logits."""
    mask = op.active
    w2 = w2_diag(logits_s, logits_t, op, eps)
    return (alpha[mask] * w2[mask]).mean()


def self_normalized(term, eps_sg=EPS_SG):
    """term / (sg(term) + eps_sg), the stop-gradient rescaling of Eq. (13)."""
    return term / (term.detach() + eps_sg)


def caw_penalty(l_h, l_z, lambda_caw, eps_sg=EPS_SG):
    """CAW term of Eq. (13): lambda_caw * L / (sg(L) + eps_sg), L = l_h + l_z."""
    return lambda_caw * self_normalized(l_h + l_z, eps_sg)
