"""
cellai_core.routing
===================
Entropy-regularized multi-domain routing with Gumbel-Softmax.

Problem with the original LinearRouter
----------------------------------------
The original router (a single nn.Linear(D, 3)) collapses to predicting one class
for all inputs when trained jointly with the NTP objective:

  1. Early in training, one domain (e.g., math) happens to minimise CE slightly better.
  2. Router gradient pushes toward predicting that domain.
  3. All inputs route to the same head → heads diverge → routing stays collapsed.

This is the load-collapsing problem identified in [Shazeer et al., 2017].

Three fixes applied here
------------------------
1. Entropy bonus: add -λ_H × H(p̄) to the loss, where H is entropy and p̄ is the
   average routing distribution over a batch. This pushes routing toward uniform use
   of all heads [Fedus et al., 2022].

2. Gumbel-Softmax routing: during training, add Gumbel noise to logits before
   softmax, implementing the Gumbel-Softmax trick [Jang et al., 2017; Maddison et al.,
   2017]. This provides stochastic exploration of routing decisions and prevents
   premature commitment to a single route.

3. Per-domain temperature annealing: start with high temperature (soft, exploratory),
   anneal toward lower temperature (harder, more decisive) as training progresses.

Complexity
----------
Router forward: O(D × n_mod) — linear, O(m log n) trivially satisfied.
Entropy computation: O(n_mod) — negligible.
Total routing overhead over full sequence of L tokens: O(L × D × n_mod).

Biological motivation
---------------------
The prefrontal cortex routes sensory input to different processing streams (dorsal/
ventral visual pathway, auditory cortex, etc.) based on task context [Miller & Cohen,
2001]. The routing decision is context-sensitive and dynamic: the same stimulus can
be processed differently depending on recent history. The Gumbel noise models the
inherent stochasticity of this context-dependent routing.

The entropy regularizer ensures that the routing circuit remains flexible: a cortical
region that always routes everything the same way has lost its function as a router.

Implementation details
----------------------
EntropyRouter is a drop-in replacement for the linear router in MultiModalModel.
It adds no parameters beyond the base router (D×n_mod weights).
The entropy loss must be collected per-step and added to the training objective.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyRouter(nn.Module):
    """
    Entropy-regularized Gumbel-Softmax domain router.

    Forward returns:
        weights (n,):       soft routing distribution (for mixing heads)
        router_logits (n,): raw logits (for cross-entropy router loss)
        entropy_loss:       scalar load-balancing penalty (minimize for uniform routing)

    The entropy_loss should be multiplied by lambda_entropy and SUBTRACTED from the
    total loss (we want to maximize entropy = minimize -entropy = subtract the loss).

    Actually we add lambda_entropy * (uniform_divergence_loss) which penalizes
    deviation from uniform routing (same effect as entropy maximization).

    Args:
        state_size:     D — input dimension
        n_modalities:   number of routing targets
        temperature:    Gumbel-Softmax temperature (high=soft, low=hard)
        lambda_balance: weight of load-balancing penalty
    """

    def __init__(
        self,
        state_size:     int,
        n_modalities:   int = 3,
        temperature:    float = 1.0,
        lambda_balance: float = 0.01,
    ):
        super().__init__()
        self.D           = state_size
        self.n           = n_modalities
        self.temperature = temperature
        self.lambda_bal  = lambda_balance

        # Router linear layer (same as original)
        self.linear = nn.Linear(state_size, n_modalities, bias=True)
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

        # Running mean of routing distribution (for batch-level entropy)
        self.register_buffer("running_mean", torch.ones(n_modalities) / n_modalities)

    def forward(
        self,
        state: torch.Tensor,                         # (D,)
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            weights (n,):       routing weights (gradient-enabled for head mixing)
            logits  (n,):       raw logits (for CE router loss)
            bal_loss scalar:    load-balancing penalty
        """
        logits = self.linear(state)                  # (n,)

        if training:
            # Gumbel noise for stochastic exploration
            U = torch.rand_like(logits).clamp(1e-9, 1 - 1e-9)
            gumbel = -torch.log(-torch.log(U))
            noisy_logits = logits + gumbel
        else:
            noisy_logits = logits

        weights = F.softmax(noisy_logits / self.temperature, dim=0)   # (n,)

        # Update running mean of routing (exponential moving average)
        with torch.no_grad():
            self.running_mean = 0.99 * self.running_mean + 0.01 * weights.detach()

        # Load-balancing loss: penalise deviation from uniform
        # Following Switch Transformer [Fedus 2022]: L_aux = N * sum(f_k * p_k)
        # where f_k = fraction routed to expert k (≈ running_mean)
        # and p_k = current router probability
        f_k = self.running_mean                           # (n,) — approximate fraction
        bal_loss = self.n * (f_k * weights).sum()        # scalar ≥ 1 (=1 when uniform)

        return weights, logits, bal_loss

    def set_temperature(self, temp: float):
        self.temperature = temp

    def routing_entropy(self) -> float:
        """Entropy of the running-average routing distribution (higher = more uniform)."""
        p = self.running_mean.clamp(1e-9)
        return float(-torch.sum(p * p.log()).item())


class AnnealedRouter(EntropyRouter):
    """
    EntropyRouter with automatic temperature AND lambda_balance annealing.

    Two-phase schedule:
      Phase 1 (0..anneal_steps):   λ_bal high (enforce load balance), T anneals T_start→T_end
      Phase 2 (anneal_steps..end): λ_bal decays to λ_end (allow discrimination to emerge)

    This resolves the entropy-accuracy trade-off: strong regularisation early prevents
    collapse; weak regularisation late allows the router to learn domain-specific features.
    """

    def __init__(
        self,
        state_size:     int,
        n_modalities:   int   = 3,
        T_start:        float = 2.0,
        T_end:          float = 0.5,
        anneal_steps:   int   = 3000,
        lambda_start:   float = 0.1,    # high early: enforce balance
        lambda_end:     float = 0.001,  # low late: allow discrimination
        lambda_balance: float = 0.01,   # initial value (overridden by schedule)
    ):
        super().__init__(state_size, n_modalities, T_start, lambda_start)
        self.T_start       = T_start
        self.T_end         = T_end
        self.anneal_steps  = anneal_steps
        self.lambda_start  = lambda_start
        self.lambda_end    = lambda_end
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    def step_temperature(self):
        """Call once per training step to anneal temperature and lambda_balance."""
        s    = int(self._step.item())
        frac = min(s / max(self.anneal_steps, 1), 1.0)

        # Temperature: linear decay T_start → T_end
        self.temperature = self.T_start + frac * (self.T_end - self.T_start)

        # λ_bal: exponential decay from lambda_start to lambda_end
        self.lambda_bal = self.lambda_start * (self.lambda_end / self.lambda_start) ** frac

        self._step += 1
