import numpy as np
from typing import List, Optional, Tuple
from .circuit import Circuit, gate_unitary, gate_unitary_2q
#do you guys love this kind of my style 

class MPS:
    def __init__(self, n: int, chi_max: int = 64, svd_cutoff: float = 1e-10,
                 rng: Optional[np.random.Generator] = None):
        self.n = n
        self.chi_max = chi_max
        self.svd_cutoff = svd_cutoff
        self.tensors: List[np.ndarray] = [np.zeros((1, 2, 1), dtype=complex) for _ in range(n)]
        for t in self.tensors:
            t[0, 0, 0] = 1.0
        self.order = list(range(n))
        self.pos_to_qubit = list(range(n))
        self.rng = rng if rng is not None else np.random.default_rng()
        self.last_discarded_weight = 0.0

    def bond_dims(self) -> List[int]:
        return [t.shape[2] for t in self.tensors[:-1]]

    def apply_1q(self, u: np.ndarray, logical_q: int):
        pos = self.order[logical_q]
        t = self.tensors[pos]
        self.tensors[pos] = np.einsum("ab,lbr->lar", u, t)

    def _apply_gate_adjacent(self, gate4: np.ndarray, pos: int):
        A = self.tensors[pos]
        B = self.tensors[pos + 1]
        Dl, _, Dm = A.shape
        _, _, Dr = B.shape
        theta_in = np.einsum("lsm,mtr->lstr", A, B)
        g = gate4.reshape(2, 2, 2, 2)
        theta4 = np.einsum("pqkm,lkmr->lpqr", g, theta_in)
        theta = theta4.reshape(Dl * 2, 2 * Dr)
        U, S, Vh = np.linalg.svd(theta, full_matrices=False)
        total_weight = np.sum(S ** 2)
        keep = min(self.chi_max, len(S))
        cum = np.cumsum((S[::-1]) ** 2)[::-1]
        significant = np.sum(S > self.svd_cutoff * (S[0] if len(S) else 1.0))
        keep = max(1, min(keep, significant))
        discarded = np.sum(S[keep:] ** 2)
        self.last_discarded_weight = float(discarded / total_weight) if total_weight > 0 else 0.0
        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]
        norm = np.linalg.norm(S)
        if norm > 0:
            S = S / norm
        self.tensors[pos] = U.reshape(Dl, 2, keep)
        self.tensors[pos + 1] = (np.diag(S) @ Vh).reshape(keep, 2, Dr)

    def _swap_adjacent_positions(self, pos: int):
        swap_u = gate_unitary_2q("SWAP")
        self._apply_gate_adjacent(swap_u, pos)
        q_at_pos, q_at_pos1 = self.pos_to_qubit[pos], self.pos_to_qubit[pos + 1]
        self.pos_to_qubit[pos], self.pos_to_qubit[pos + 1] = q_at_pos1, q_at_pos
        self.order[q_at_pos], self.order[q_at_pos1] = pos + 1, pos
#i luv irrelevant comments bleh what the helly welly banana jelly
    def apply_2q(self, gate4: np.ndarray, logical_a: int, logical_b: int):
        pos_a = self.order[logical_a]
        pos_b = self.order[logical_b]
        if pos_a > pos_b:
            pos_a, pos_b = pos_b, pos_a
            gate4 = self._swap_gate_operands(gate4)
        while pos_b - pos_a > 1:
            self._swap_adjacent_positions(pos_a)
            pos_a += 1
        self._apply_gate_adjacent(gate4, pos_a)
        self._recanonicalize()

    @staticmethod
    def _swap_gate_operands(gate4: np.ndarray) -> np.ndarray:
        g = gate4.reshape(2, 2, 2, 2)
        g = np.transpose(g, (1, 0, 3, 2))
        return g.reshape(4, 4)

    def _recanonicalize(self):
        n = self.n
        for i in range(n - 1):
            t = self.tensors[i]
            Dl, s, Dr = t.shape
            mat = t.reshape(Dl * s, Dr)
            Q, R = np.linalg.qr(mat)
            newDr = Q.shape[1]
            self.tensors[i] = Q.reshape(Dl, s, newDr)
            self.tensors[i + 1] = np.einsum("ab,btr->atr", R, self.tensors[i + 1])
        norm = np.linalg.norm(self.tensors[n - 1].reshape(-1))
        if norm > 0:
            self.tensors[n - 1] = self.tensors[n - 1] / norm

    def normalize(self):
        self._recanonicalize()

    def to_statevector(self) -> np.ndarray:
        if self.n > 20:
            raise ValueError("to_statevector is exponential; refusing for n>20")
        vec = self.tensors[0]
        for t in self.tensors[1:]:
            vec = np.tensordot(vec, t, axes=([-1], [0]))
        vec = vec[0, ..., 0]
        vec = np.transpose(vec, self.order)
        vec = vec.reshape(-1)
        return vec / np.linalg.norm(vec)

    def schmidt_values(self, cut: int) -> np.ndarray:
        n = self.n
        tensors = [self.tensors[i].copy() for i in range(n)]
        for i in range(cut + 1):
            t = tensors[i]
            Dl, s, Dr = t.shape
            Q, R = np.linalg.qr(t.reshape(Dl * s, Dr))
            tensors[i] = Q.reshape(Dl, s, Q.shape[1])
            if i + 1 < n:
                tensors[i + 1] = np.einsum("ab,btr->atr", R, tensors[i + 1])
        for i in range(n - 1, cut + 1, -1):
            t = tensors[i]
            Dl, s, Dr = t.shape
            mat = t.reshape(Dl, s * Dr)
            Q, L = np.linalg.qr(mat.T)
            Q = Q.T
            L = L.T
            tensors[i] = Q.reshape(Q.shape[0], s, Dr)
            tensors[i - 1] = np.einsum("lsm,mr->lsr", tensors[i - 1], L)
        A = tensors[cut]
        B = tensors[cut + 1]
        Dl, s, Dc = A.shape
        _, s2, Dr = B.shape
        theta = np.einsum("lsm,mtr->lstr", A, B).reshape(Dl * s, s2 * Dr)
        S = np.linalg.svd(theta, compute_uv=False)
        p = (S ** 2)
        p = p / np.sum(p)
        return p

    def entanglement_entropy(self, cut: int) -> float:
        p = self.schmidt_values(cut)
        p = p[p > 1e-14]
        return float(-np.sum(p * np.log2(p)))

    def sample_bitstring(self) -> Tuple[int, ...]:
        n = self.n
        self._recanonicalize()
        tensors = [self.tensors[i].copy() for i in range(n)]
        vec = None
        left_env = np.ones((1, 1), dtype=complex)
        outcome = [0] * n
        for pos in range(n):
            t = np.einsum("ab,bsr->asr", left_env, tensors[pos])
            probs = np.array([np.sum(np.abs(t[:, s, :]) ** 2) for s in range(2)]).real
            probs = probs / probs.sum()
            s_out = 0 if self.rng.random() < probs[0] else 1
            outcome[self.pos_to_qubit[pos]] = s_out
            slice_t = t[:, s_out, :]
            nrm = np.linalg.norm(slice_t)
            left_env = slice_t / nrm
        return tuple(outcome)

    def run(self, circuit: Circuit) -> List[Optional[int]]:
        results: List[Optional[int]] = []
        for g in circuit.gates:
            if g.name == "MEASURE":
                self._recanonicalize()
                results.append(None)
            elif len(g.qubits) == 1:
                self.apply_1q(gate_unitary(g.name, g.params), g.qubits[0])
            else:
                self.apply_2q(gate_unitary_2q(g.name), g.qubits[0], g.qubits[1])
        self._recanonicalize()
        return results
