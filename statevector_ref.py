import numpy as np
from typing import List, Optional
from .circuit import Circuit, gate_unitary, gate_unitary_2q

#why did the chicken cross the road?
#to get to the other side!
class StatevectorSim:
    def __init__(self, n: int, rng: Optional[np.random.Generator] = None):
        if n > 24:
            raise ValueError("StatevectorSim is exponential; use tableau or MPS for n>24")
        self.n = n
        self.state = np.zeros(2 ** n, dtype=complex)
        self.state[0] = 1.0
        self.rng = rng if rng is not None else np.random.default_rng()

    def _apply_1q(self, u: np.ndarray, q: int):
        n = self.n
        psi = self.state.reshape([2] * n)
        psi = np.moveaxis(psi, q, 0)
        psi = np.tensordot(u, psi, axes=([1], [0]))
        psi = np.moveaxis(psi, 0, q)
        self.state = psi.reshape(-1)

    def _apply_2q(self, u4: np.ndarray, q0: int, q1: int):
        n = self.n
        u = u4.reshape(2, 2, 2, 2)
        psi = self.state.reshape([2] * n)
        psi = np.moveaxis(psi, [q0, q1], [0, 1])
        psi = np.tensordot(u, psi, axes=([2, 3], [0, 1]))
        psi = np.moveaxis(psi, [0, 1], [q0, q1])
        self.state = psi.reshape(-1)

    def measure(self, q: int) -> int:
        n = self.n
        psi = self.state.reshape([2] * n)
        psi = np.moveaxis(psi, q, 0)
        p0 = np.sum(np.abs(psi[0]) ** 2).real
        outcome = 0 if self.rng.random() < p0 else 1
        if outcome == 0:
            psi[1] = 0
            psi = psi / np.sqrt(p0)
        else:
            psi[0] = 0
            psi = psi / np.sqrt(1 - p0)
        psi = np.moveaxis(psi, 0, q)
        self.state = psi.reshape(-1)
        return outcome

    def run(self, circuit: Circuit) -> List[Optional[int]]:
        results: List[Optional[int]] = []
        for g in circuit.gates:
            if g.name == "MEASURE":
                results.append(self.measure(g.qubits[0]))
            elif len(g.qubits) == 1:
                self._apply_1q(gate_unitary(g.name, g.params), g.qubits[0])
            else:
                self._apply_2q(gate_unitary_2q(g.name), g.qubits[0], g.qubits[1])
        return results

    def probabilities(self) -> np.ndarray:
        return np.abs(self.state) ** 2

    def fidelity(self, other: np.ndarray) -> float:
        return float(np.abs(np.vdot(self.state, other)) ** 2)
