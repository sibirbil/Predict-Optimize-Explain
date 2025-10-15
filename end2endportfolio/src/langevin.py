import jax
import jax.numpy as jnp
import jax.random as random
from jax.tree_util import tree_map, tree_unflatten, tree_structure, tree_leaves, tree_reduce
from functools import partial
from numbers import Number
from typing import Tuple, List, Optional
import torch

# -----------------------
# Example scalar function
# -----------------------
def F(x):
    return (x**4)/10 + (x**3)/10 - (x**2)

F_grad = jax.grad(lambda x: jnp.reshape(F(x), ()))

hyps = (F, F_grad, 0.1)
state0 = (random.PRNGKey(0), 0.0)
state1 = (random.PRNGKey(1), jnp.array(0.))
state2 = (random.PRNGKey(2), jnp.array([0.]))

# ---------- Helpers ----------
def _parse_hyps(hyps_tuple):
    """Parse (func, grad_func, eta[, beta][, clip_low, clip_high])."""

    if len(hyps_tuple) < 3:
        raise ValueError("hyps must contain at least (func, grad_func, eta)")

    func, grad_func, eta = hyps_tuple[:3]
    beta = 1.0
    clip_low = clip_high = None

    rest = hyps_tuple[3:]
    if rest:
        first = rest[0]
        if isinstance(first, Number):
            beta = float(first)
            rest = rest[1:]
        if rest:
            clip_low = rest[0]
            if len(rest) > 1:
                clip_high = rest[1]

    return func, grad_func, eta, beta, (clip_low, clip_high)

# ==============================
# JAX: single-leaf Langevin step
# ==============================
def leaf_langevin(
    x       : jax.Array,
    g       : jax.Array,
    xi      : jax.Array,
    eta     : float,
    clip_to : Tuple[Optional[float], Optional[float]],
):
    step = x - eta * g + jnp.sqrt(2 * eta) * xi
    a, b = clip_to
    # jnp.clip does not accept None; handle individually
    if a is not None:
        step = jnp.maximum(step, a)
    if b is not None:
        step = jnp.minimum(step, b)
    return step

def langevin_step(
    key,            # PRNGKey
    x,              # PyTree of arrays
    g,              # same PyTree structure as x
    eta: float,     # step size
    clip_to: Tuple[Optional[float], Optional[float]] = (None, None),
):
    """
    Calculates the next step in Langevin dynamics for PyTree inputs.
    Returns (x_next, xi) where xi has same PyTree structure as x.
    """
    keys = random.split(key, num=len(tree_leaves(x)))
    keys_tree = tree_unflatten(tree_structure(x), keys)

    xi = tree_map(lambda k, leaf: random.normal(k, shape=leaf.shape), keys_tree, x)
    leaf_step = partial(leaf_langevin, eta=eta, clip_to=clip_to)
    x_next = tree_map(leaf_step, x, g, xi)
    return x_next, xi

def MALA_step(state, hyps):
    """
    One MALA step for PyTree inputs.
    """
    key, x = state
    func, grad_func, eta, beta, clip_to = _parse_hyps(hyps)
    g = grad_func(x)
    g = tree_map(lambda leaf: beta * leaf, g)

    key, accept_key = random.split(key)
    x_proposed, xi = langevin_step(key, x, g, eta, clip_to)

    g_proposed = grad_func(x_proposed)
    g_proposed = tree_map(lambda leaf: beta * leaf, g_proposed)

    def leaf_log_proposal_ratio(x_leaf, x_prop_leaf, g_prop_leaf, xi_leaf):
        forward = -jnp.sum(jnp.square(xi_leaf)) / (4 * eta)
        reverse = -jnp.sum(jnp.square(x_leaf - x_prop_leaf + eta * g_prop_leaf)) / (4 * eta)
        return reverse - forward

    log_prop_ratio = tree_reduce(lambda a, b: a + b,
                                 tree_map(leaf_log_proposal_ratio, x, x_proposed, g_proposed, xi))

    log_acc_ratio = -beta * func(x_proposed) + beta * func(x) + log_prop_ratio
    acc_prob = jnp.minimum(1.0, jnp.exp(log_acc_ratio))

    u = random.uniform(accept_key)
    accepted = u < acc_prob
    x_next = tree_map(lambda xp, xc: jnp.where(accepted, xp, xc), x_proposed, x)
    return key, x_next

from src.utils import as_scheduler

def MALA_chain(state, hyps, NSteps: int):
    func, grad_func, eta, beta, clip_to = _parse_hyps(hyps)
    eta = as_scheduler(eta)
    # carry = (key, x, step)
    key, x = state
    carry_init = (key, x, 0)

    def f(carry, _):
        key, x, step = carry
        lr = eta(step)
        new_hyps = (func, grad_func, lr, beta, *clip_to)
        next_key, x_next = MALA_step((key, x), new_hyps)
        return (next_key, x_next, step + 1), x_next

    (last_key, last_x, _), x_traj = jax.lax.scan(f, carry_init, None, length=NSteps)
    return (last_key, last_x), x_traj

def nt_MALA(state, hyps, Nsteps: int):
    func, grad_func, eta, beta, clip_to = _parse_hyps(hyps)
    eta = as_scheduler(eta)
    key, x = state

    outs = []
    for i in range(Nsteps):
        lr = eta(i)
        new_hyps = (func, grad_func, lr, beta, *clip_to)
        key, x = MALA_step((key, x), new_hyps)
        outs.append(x)

    # jax.tree.map doesn't exist; use tree_map + stack via unpack
    return (key, x), tree_map(lambda *xs: jnp.stack(xs), *outs)

# ------------------------------------
# TORCH: single-tensor MALA (no PyTrees)
# ------------------------------------
def torch_langevin_step(
    x: torch.Tensor,
    g: torch.Tensor,
    eta: float,
    clip_to: Tuple[Optional[float], Optional[float]] = (None, None),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Langevin update for a single tensor. Returns (x_next, xi).
    """
    xi = torch.randn_like(x)
    two_eta = torch.as_tensor(2.0 * eta, dtype=x.dtype, device=x.device)
    x_next = x - eta * g + torch.sqrt(two_eta) * xi

    a, b = clip_to
    if a is not None or b is not None:
        # torch.clamp accepts None for min/max; ensure same dtype/device when not None
        min_v = torch.as_tensor(a, dtype=x.dtype, device=x.device) if a is not None else None
        max_v = torch.as_tensor(b, dtype=x.dtype, device=x.device) if b is not None else None
        x_next = torch.clamp(x_next, min=min_v, max=max_v)

    return x_next.detach(), xi

def torch_MALA_step(x: torch.Tensor, hyps) -> torch.Tensor:
    """
    One MALA step for a single tensor with MH accept/reject.
    hyps: (func, grad_func, eta[, beta][, clip_low, clip_high])
    """
    func, grad_func, eta, beta, clip_to = _parse_hyps(hyps)

    # gradient at current position
    g = grad_func(x)
    g_beta = beta * g

    # propose
    x_prop, xi = torch_langevin_step(x, g_beta, eta, clip_to)

    # gradient at proposal
    g_prop = grad_func(x_prop.requires_grad_())
    g_prop_beta = beta * g_prop

    # log proposal ratio
    forward = -torch.sum(xi ** 2) / (4 * eta)
    reverse = -torch.sum((x - x_prop + eta * g_prop_beta) ** 2) / (4 * eta)
    log_prop_ratio = reverse - forward

    # acceptance
    log_acc_ratio = -beta * func(x_prop) + beta * func(x) + log_prop_ratio
    one = torch.ones((), dtype=log_acc_ratio.dtype, device=log_acc_ratio.device)
    acc_prob = torch.minimum(one, torch.exp(log_acc_ratio))

    accepted = torch.rand((), device=acc_prob.device).item() < acc_prob.item()
    x_next = x_prop if accepted else x
    return x_next

def torch_MALA_chain(
    x: torch.Tensor,
    hyps,
    NSteps: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MALA chain for a single tensor.
    Returns (final_x, stacked_trajectory) where stacked_trajectory has shape (NSteps, *x.shape).
    """
    func, grad_func, eta, beta, clip_to = _parse_hyps(hyps)
    eta = as_scheduler(eta)

    x.requires_grad_(True)
    traj: List[torch.Tensor] = []

    for step in range(NSteps):
        lr = eta(step)
        new_hyps = (func, grad_func, lr, beta, *clip_to)
        x = torch_MALA_step(x, new_hyps)
        traj.append(x.detach())

    return x.detach(), torch.stack(traj)
