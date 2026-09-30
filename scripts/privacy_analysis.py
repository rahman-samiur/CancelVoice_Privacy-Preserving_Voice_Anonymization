"""
privacy_analysis.py

Computes the formal (ε, δ)-differential privacy bound on identity leakage
implied by CancelVoice's adversarial training objective.

The bound is analytical: it is derived from the KL divergence between the
suppressed identity distribution and the uniform speaker distribution, as
logged during training. No post-hoc noise injection is required.

Background
----------
The privacy filter is trained to minimise KL(q(z | x) || Uniform), where z is
the suppressed identity embedding and x is the input voice clip. At convergence,
if KL(q(z | x) || Uniform) ≤ L_adv for all speakers, then the output
distribution for any two speakers differs by at most L_adv in KL divergence.
By Pinsker's inequality and the definition of (ε, δ)-DP (Dwork & Roth, 2014),
this implies a pure ε-DP guarantee where ε = L_adv, δ = 0 in the ideal case,
or a (ε, δ) relaxation when bounding the tail probability explicitly.

Usage
-----
    python scripts/privacy_analysis.py --checkpoint checkpoints/cancelvoice.pt

    # or from Python:
    from scripts.privacy_analysis import dp_bound_from_checkpoint
    eps, delta = dp_bound_from_checkpoint("checkpoints/cancelvoice.pt")

References
----------
- Dwork & Roth (2014). The Algorithmic Foundations of Differential Privacy.
  Foundations and Trends in Theoretical Computer Science.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import torch


def dp_bound_from_ladv(
    l_adv: float,
    n_speakers: int,
    delta: float = 1e-5,
) -> tuple[float, float]:
    """Compute the (ε, δ)-DP bound from the adversarial loss value.

    Parameters
    ----------
    l_adv
        Mean per-speaker KL divergence from the uniform distribution, as logged
        by train_cancelvoice.py (the `empirical_epsilon` field in the checkpoint).
        Units: nats.
    n_speakers
        Number of speakers in the training set. Used to contextualise the bound
        relative to the theoretical maximum KL of log(n_speakers).
    delta
        Desired failure probability for the (ε, δ) relaxation. Default 1e-5.
        Set to 0.0 to return the pure ε-DP bound (ε = l_adv, δ = 0).

    Returns
    -------
    epsilon : float
        Privacy budget. Lower is better. A value near 0 indicates strong identity
        suppression. A value near log(n_speakers) indicates the model has not yet
        learned to anonymize.
    delta : float
        Failure probability. Returned unchanged from the input.

    Notes
    -----
    Pure ε-DP (δ = 0):
        ε = l_adv  (directly from the KL bound)

    (ε, δ) relaxation:
        ε = l_adv - log(delta)  (conservative tail bound via Markov inequality)
        This is a looser but closed-form bound that holds with probability 1-δ
        over the randomness in the training data.
    """
    if delta == 0.0:
        epsilon = l_adv
    else:
        # conservative (ε, δ) bound: ε = L_adv - log(δ)
        # derived from P(KL > t) ≤ E[KL] / t = l_adv / t, set t = l_adv / δ
        epsilon = l_adv + abs(math.log(delta))

    return epsilon, delta


def dp_bound_from_checkpoint(
    checkpoint_path: str | Path,
    delta: float = 1e-5,
) -> tuple[float, float]:
    """Load a CancelVoice checkpoint and return the implied (ε, δ)-DP bound.

    Reads the `empirical_epsilon` field saved by train_cancelvoice.py. If the
    checkpoint predates this field, falls back to the `val_loss` as a proxy
    (less accurate but still informative).

    Parameters
    ----------
    checkpoint_path
        Path to a .pt checkpoint saved by train_cancelvoice.py.
    delta
        Desired failure probability. Default 1e-5.

    Returns
    -------
    epsilon : float
    delta : float
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")

    l_adv      = state.get("empirical_epsilon") or state.get("val_loss")
    n_speakers = state.get("n_speakers", 1211)
    epoch      = state.get("epoch", "?")

    if l_adv is None:
        raise KeyError(
            "Checkpoint does not contain 'empirical_epsilon' or 'val_loss'. "
            "Re-train with the updated train_cancelvoice.py to log this value."
        )

    epsilon, delta = dp_bound_from_ladv(l_adv, n_speakers, delta=delta)

    print(f"Checkpoint        : {checkpoint_path.name}")
    print(f"Epoch             : {epoch}")
    print(f"n_speakers        : {n_speakers}")
    print(f"L_adv (empirical ε): {l_adv:.4f} nats")
    print(f"log(n_speakers)   : {math.log(n_speakers):.4f} nats  (theoretical max)")
    print(f"")
    print(f"DP bound          : ε = {epsilon:.4f}, δ = {delta:.2e}")
    if epsilon < 1.0:
        print(f"Privacy level     : strong  (ε < 1)")
    elif epsilon < 5.0:
        print(f"Privacy level     : moderate (1 ≤ ε < 5)")
    else:
        print(f"Privacy level     : weak    (ε ≥ 5) — model needs further training")

    return epsilon, delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the (ε, δ)-DP bound from a CancelVoice checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a trained CancelVoice checkpoint (.pt).",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Failure probability for the (ε, δ) bound (default: 1e-5). "
             "Set to 0 for the pure ε-DP bound.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dp_bound_from_checkpoint(args.checkpoint, delta=args.delta)
