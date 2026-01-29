import jax
import jax.numpy as jnp
import jax.random as random
from jax.tree_util import tree_map, tree_unflatten, tree_structure, tree_leaves, tree_reduce
from functools import partial
from numbers import Number
from typing import Tuple, List, Optional
import torch
from end2endportfolio.src.utils import as_scheduler


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
    if isinstance(a, torch.Tensor) and (b is None):
        x_next = torch.max(x_next, a)
    elif isinstance(b, torch.Tensor) and (a is None):
        x_next = torch.min(x_next, b)
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        x_next = torch.min(torch.max(x_next, a), b)
    elif (a is not None) or (b is not None):
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
    with torch.no_grad():
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



####################################################
### numpy implementation for locally const funcs ###
####################################################

import numpy as np
from typing import NamedTuple, Callable, Union

Array = np.ndarray

class Hyperparameters(NamedTuple):
    func            : Callable[[Array], Array]  
    # func is a scalar valued function to apply MALA to,
    # with signature will be (n, *x.shape)-> (n,) (i.e. batchable)
    beta            : float                     # inverse temperature
    eta             : float                     # initial step size
    # if grad is not given we will approximate 
    # the gradient of gaussian smoothed func.
    grad_func       : Optional[Callable]  = None 
    reg             : Optional[Callable]  = None
    # target_rate alters step size so that the MALA acceptance rate is within 
    # 0.8 and 1.2 of the target_rate. If target_rate==None then constant step size
    target_rate     : Optional[float] = 0.574    
    # target_change alters smoothing rate so that MALA changes leaf within 
    # 0.8 and 1.2 percent of target_change. If None then smoothing is constant
    target_change   : Optional[float] = 0.1   
    check_every     : int = 100
    n_samples       : Optional[int] = 1         # number of samples to take for grad comp
    sigma           : Optional[float] = 0.1     # sigma of the noise
    

def convolve_and_diff(
        x           : Array,
        n_samples   : int,
        sigma       : float,
        func        : Callable,
        grad_func   : Optional[Callable]
    ):
    """
    Get an estimate for the derivative of E[func], the expectation of func,
    with respect to the mean of the distribution. The distribution is a 
    normal distribution with fixed variance sigma^2. This is equal to the
    correlation of F and gaussian noise itself, i.e. ∫F(x + ε)ε dε, which
    is estimated by random Gaussian samples, making sure -/+ ε are included
    (antithetic sampling), to reduce variance in the estimation.
    """
    x_size = x[0].shape
    if grad_func is None:
        noise_base = sigma * np.random.randn(n_samples, *x_size)
        x_pos = x + noise_base    # (n_samples, *x_size)
        x_neg = x - noise_base    # (n_samples, *x_size)

        T_pos = func(x_pos)
        T_neg = func(x_neg)
        # Below approximates the gradient of ∫F(x + σε)ε dε for F = func
        # where ε ~ N(0,I). Better behaved than ∇F when F is a step function.
        g = (T_pos - T_neg) @ noise_base/(2*sigma*n_samples)
        g = g[None,:]

    else:
        g = grad_func(x)

    return g

def np_MALA_step(
        x       : Array, 
        hyps    : Hyperparameters, 
        eta     : Optional[float] = None,
        sigma   : Optional[float] = None
    ):

    if eta is None:
        eta = hyps.eta/hyps.beta
    if sigma is None:
        sigma = hyps.sigma
    
    g = convolve_and_diff(x, hyps.n_samples, sigma, hyps.func, hyps.grad_func)
    g *= hyps.beta

    # Generate noise with same shape as x
    xi = np.random.randn(*x.shape)
    
    # Apply Langevin update: x - eta*g + sqrt(2*eta)*xi
    update = -eta*g + np.sqrt(2*eta)*xi
    x_proposed = x + update
    # Acceptance probability assuming ∇F(x) = ∇F(x') which may be the case 
    # when ∇F is the gradient of the gaussian convolution of a step function, 
    # since the convolution is flat within leaves or mostly linear in transitions. 
    # log q(x|x') - log q(x'|x) = (∇F)^T (-η(∇F) + sqrt(2η)ξ) which is computed
    # by noting that x' = x - η(∇F(x)) + sqrt(2η)ξ and to go back we would need
    # x = x' - η(∇F(x')) + sqrt(2η)ξ', for a particular ξ'. Which ends up equaling 
    # ξ' = sqrt(2η)∇F(x) - ξ. Then we can calculate the log transition density as 
    # q(x|x') = -(1/2)||ξ'||^2. The forward transition log density is similarly 
    # q(x'|x) = -(1/2)||ξ||^2. Subtracting we get the cancellation and the result.
    log_acc_ratio = -hyps.beta*hyps.func(x_proposed) + hyps.beta*hyps.func(x) + np.dot(g, update[0]) 
    if hyps.reg is not None:
        log_acc_ratio += -hyps.reg(x_proposed) + hyps.reg(x)              
    acc_prob = np.exp(np.minimum(0., log_acc_ratio))
    
    # Generate random uniform value for acceptance decision
    uniform_sample = np.random.rand()
    accepted = (uniform_sample < acc_prob).squeeze()
    
    # Accept or reject proposal
    x_next = x_proposed if accepted else x
    
    changed = (hyps.func(x_next) != hyps.func(x))
    changed = changed.squeeze()
    
    return x_next, accepted, changed


def adaptive_np_MALA(
    x       : Array,
    hyps    : Hyperparameters,
    Nsteps  : int
    ):
    acc = 0     # number of accepted steps
    chg = 0     # number of steps where leaf changed
    tot = 0     # total steps taken
    eta = hyps.eta/hyps.beta
    sigma = hyps.sigma
    traj = [x.squeeze()]
    for step in range(Nsteps):
        x, accepted, changed = np_MALA_step(x, hyps, eta, sigma)
        acc += accepted     # accepted is boolean and is converted to 0 or 1.
        chg += changed      # changed is also a boolean
        tot += 1
        traj.append(x.squeeze())
        if step % hyps.check_every == hyps.check_every-1:
            eta, sigma = adapt_parameters(acc, chg, tot, eta, sigma, hyps)
            print(f"On step {step} we have η={eta*hyps.beta:.6f} and σ={sigma:.4f}")
            acc = 0
            chg = 0
            tot = 0     #reset the counters

            
    return x, np.array(traj)

def adapt_parameters(acc, chg, tot, eta, sigma, hyps: Hyperparameters):

    if hyps.target_rate is None:
        pass        # eta stays fixed
    elif acc/tot > hyps.target_rate*1.2:
        eta = min(1.1*eta, 1./hyps.beta)
    elif acc/tot < hyps.target_rate*0.8:
        eta = max(0.9*eta, 1e-3/hyps.beta)
            
    if hyps.target_change is None:
        pass        # sigma stays fixed
    elif chg/tot > hyps.target_change*2:
        sigma = max(0.9*sigma, 1e-4)   # smaller sigma decreases horizon
    elif chg/tot < hyps.target_change/2:
        sigma = min(1.1*sigma, 1.)      # higher sigma increases horizon
    
    print(f"In the last {hyps.check_every} steps {acc} accepted, {chg} times leaf changed")

    return eta, sigma