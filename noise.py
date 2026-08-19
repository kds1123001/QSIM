import numpy as np
from typing import List, Tuple, Optional
from .tableau import StabilizerTableau
from .mps import MPS
from .circuit import gate_unitary


def pauli_channel_apply(tab: StabilizerTableau, q: int, p_x: float, p_y: float, p_z: float,
                         rng: Optional[np.random.Generator] = None):
    rng = rng if rng is not None else tab.rng
    r = rng.random()
    if r < p_x:
        tab.x_gate(q)
    elif r < p_x + p_y:
        tab.y_gate(q)
    elif r < p_x + p_y + p_z:
        tab.z_gate(q)


def depolarizing_pauli_probs(p: float) -> Tuple[float, float, float]:
    return (p / 3, p / 3, p / 3)


KRAUS_LIBRARY = {
    "bit_flip": lambda p: [
        np.sqrt(1 - p) * np.eye(2, dtype=complex),
        np.sqrt(p) * np.array([[0, 1], [1, 0]], dtype=complex),
    ],
    "phase_flip": lambda p: [
        np.sqrt(1 - p) * np.eye(2, dtype=complex),
        np.sqrt(p) * np.array([[1, 0], [0, -1]], dtype=complex),
    ],
    "depolarizing": lambda p: [
        np.sqrt(1 - p) * np.eye(2, dtype=complex),
        np.sqrt(p / 3) * np.array([[0, 1], [1, 0]], dtype=complex),
        np.sqrt(p / 3) * np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.sqrt(p / 3) * np.array([[1, 0], [0, -1]], dtype=complex),
    ],
    "amplitude_damping": lambda p: [
        np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex),
        np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex),
    ],
}


def kraus_trajectory_apply(mps: MPS, q: int, channel: str, p: float,
                            rng: Optional[np.random.Generator] = None):
    rng = rng if rng is not None else mps.rng
    kraus_ops = KRAUS_LIBRARY[channel](p)
    pos = mps.order[q]
    t = mps.tensors[pos]
    probs = []
    candidates = []
    for k in kraus_ops:
        applied = np.einsum("ab,lbr->lar", k, t)
        weight = float(np.sum(np.abs(applied) ** 2).real)
        probs.append(weight)
        candidates.append(applied)
    probs = np.array(probs)
    total = probs.sum()
    probs = probs / total if total > 0 else probs
    choice = rng.choice(len(kraus_ops), p=probs)
    chosen = candidates[choice]
    nrm = np.linalg.norm(chosen.reshape(-1))
    mps.tensors[pos] = chosen / nrm
