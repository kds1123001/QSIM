import numpy as np
from typing import Optional, Tuple, List
from .circuit import Circuit

#no comment here lol bleh 
def _g(x1, z1, x2, z2):
    if x1 == 0 and z1 == 0:
        return 0
    if x1 == 1 and z1 == 1:
        return int(z2) - int(x2)
    if x1 == 1 and z1 == 0:
        return int(z2) * (2 * int(x2) - 1)
    return int(x2) * (1 - 2 * int(z2))


class StabilizerTableau:
    def __init__(self, n: int, rng: Optional[np.random.Generator] = None):
        self.n = n
        self.x = np.zeros((2 * n + 1, n), dtype=np.uint8)
        self.z = np.zeros((2 * n + 1, n), dtype=np.uint8)
        self.r = np.zeros(2 * n + 1, dtype=np.uint8)
        for i in range(n):
            self.x[i, i] = 1
            self.z[n + i, i] = 1
        self.rng = rng if rng is not None else np.random.default_rng()

    def copy(self) -> "StabilizerTableau":
        t = StabilizerTableau(self.n, self.rng)
        t.x = self.x.copy()
        t.z = self.z.copy()
        t.r = self.r.copy()
        return t

    def h(self, q: int):
        self.r ^= self.x[:, q] & self.z[:, q]
        self.x[:, q], self.z[:, q] = self.z[:, q].copy(), self.x[:, q].copy()

    def s(self, q: int):
        self.r ^= self.x[:, q] & self.z[:, q]
        self.z[:, q] ^= self.x[:, q]

    def sdg(self, q: int):
        self.s(q); self.s(q); self.s(q)

    def x_gate(self, q: int):
        self.r ^= self.z[:, q]

    def y_gate(self, q: int):
        self.r ^= self.x[:, q] ^ self.z[:, q]

    def z_gate(self, q: int):
        self.r ^= self.x[:, q]

    def cnot(self, c: int, t: int):
        self.r ^= self.x[:, c] & self.z[:, t] & (self.x[:, t] ^ self.z[:, c] ^ 1)
        self.x[:, t] ^= self.x[:, c]
        self.z[:, c] ^= self.z[:, t]

    def cz(self, c: int, t: int):
        self.h(t); self.cnot(c, t); self.h(t)

    def swap(self, a: int, b: int):
        self.cnot(a, b); self.cnot(b, a); self.cnot(a, b)

    def _rowsum(self, h: int, i: int):
        n = self.n
        acc = 2 * int(self.r[h]) + 2 * int(self.r[i])
        for j in range(n):
            acc += _g(self.x[i, j], self.z[i, j], self.x[h, j], self.z[h, j])
        acc %= 4
        if acc == 0:
            self.r[h] = 0
        elif acc == 2:
            self.r[h] = 1
        else:
            raise RuntimeError("invalid tableau state (non-Hermitian phase)")
        self.x[h, :] ^= self.x[i, :]
        self.z[h, :] ^= self.z[i, :]

    def measure(self, a: int) -> int:
        n = self.n
        p = None
        for row in range(n, 2 * n):
            if self.x[row, a] == 1:
                p = row
                break
        if p is not None:
            for row in range(2 * n):
                if row != p and self.x[row, a] == 1:
                    self._rowsum(row, p)
            self.x[p - n, :] = self.x[p, :]
            self.z[p - n, :] = self.z[p, :]
            self.r[p - n] = self.r[p]
            self.x[p, :] = 0
            self.z[p, :] = 0
            self.z[p, a] = 1
            outcome = int(self.rng.integers(0, 2))
            self.r[p] = outcome
            return outcome
        else:
            scratch = 2 * n
            self.x[scratch, :] = 0
            self.z[scratch, :] = 0
            self.r[scratch] = 0
            for i in range(n):
                if self.x[i, a] == 1:
                    self._rowsum(scratch, n + i)
            return int(self.r[scratch])

    def apply_pauli(self, pauli: str, q: int):
        {"I": lambda: None, "X": lambda: self.x_gate(q),
         "Y": lambda: self.y_gate(q), "Z": lambda: self.z_gate(q)}[pauli]()

    def run(self, circuit: Circuit) -> List[Optional[int]]:
        results: List[Optional[int]] = []
        dispatch = {
            "H": self.h, "S": self.s, "SDG": self.sdg,
            "X": self.x_gate, "Y": self.y_gate, "Z": self.z_gate, "I": lambda q: None,
        }
        for g in circuit.gates:
            if g.name == "MEASURE":
                results.append(self.measure(g.qubits[0]))
            elif g.name == "CNOT":
                self.cnot(*g.qubits)
            elif g.name == "CZ":
                self.cz(*g.qubits)
            elif g.name == "SWAP":
                self.swap(*g.qubits)
            elif g.name in dispatch:
                dispatch[g.name](g.qubits[0])
            else:
                raise ValueError(f"non-Clifford gate {g.name} unsupported by stabilizer tableau")
        return results

    def stabilizers(self) -> List[str]:
        n = self.n
        out = []
        for row in range(n, 2 * n):
            s = "-" if self.r[row] else "+"
            for j in range(n):
                xb, zb = self.x[row, j], self.z[row, j]
                s += {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}[(xb, zb)]
            out.append(s)
        return out

    def to_statevector(self) -> np.ndarray:
        if self.n > 14:
            raise ValueError("to_statevector is exponential; refusing for n>14")
        from .statevector_ref import StatevectorSim
        stabs = self.stabilizers()
        n = self.n
        sv = StatevectorSim(n)
        psi = sv.state
        paulis = {
            "I": np.eye(2, dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        }
        dim = 2 ** n
        proj = np.eye(dim, dtype=complex)
        for s in stabs:
            sign = -1 if s[0] == "-" else 1
            ops = [paulis[c] for c in s[1:]]
            full = ops[0]
            for op in ops[1:]:
                full = np.kron(full, op)
            p = 0.5 * (np.eye(dim, dtype=complex) + sign * full)
            proj = proj @ p
        vals, vecs = np.linalg.eigh(proj @ proj.conj().T)
        idx = np.argmax(vals)
        vec = vecs[:, idx]
        vec = vec / np.linalg.norm(vec)
        return vec
