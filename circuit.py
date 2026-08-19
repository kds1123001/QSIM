from typing import List, Tuple, Optional
import numpy as np

CLIFFORD_1Q = {"H", "S", "SDG", "X", "Y", "Z", "I"}
CLIFFORD_2Q = {"CNOT", "CZ", "SWAP"}
NONCLIFFORD_1Q = {"T", "TDG", "RX", "RY", "RZ", "U1", "U3"}

class Gate:
    def __init__(self, name: str, qubits: Tuple[int, ...], params: Tuple[float, ...] = ()):
        self.name = name
        self.qubits = qubits
        self.params = params

    def is_clifford(self) -> bool:
        return self.name in CLIFFORD_1Q or self.name in CLIFFORD_2Q

    def is_measurement(self) -> bool:
        return self.name == "MEASURE"

    def __repr__(self):
        return f"Gate({self.name!r}, {self.qubits!r}, {self.params!r})"

class Circuit:
    def __init__(self, n_qubits: int, gates: Optional[List[Gate]] = None):
        self.n_qubits = n_qubits
        self.gates: List[Gate] = gates if gates is not None else []

    def h(self, q): self.gates.append(Gate("H", (q,))); return self
    def s(self, q): self.gates.append(Gate("S", (q,))); return self
    def sdg(self, q): self.gates.append(Gate("SDG", (q,))); return self
    def x(self, q): self.gates.append(Gate("X", (q,))); return self
    def y(self, q): self.gates.append(Gate("Y", (q,))); return self
    def z(self, q): self.gates.append(Gate("Z", (q,))); return self
    def t(self, q): self.gates.append(Gate("T", (q,))); return self
    def tdg(self, q): self.gates.append(Gate("TDG", (q,))); return self
    def rx(self, q, theta): self.gates.append(Gate("RX", (q,), (theta,))); return self
    def ry(self, q, theta): self.gates.append(Gate("RY", (q,), (theta,))); return self
    def rz(self, q, theta): self.gates.append(Gate("RZ", (q,), (theta,))); return self
    def cnot(self, c, t): self.gates.append(Gate("CNOT", (c, t))); return self
    def cz(self, c, t): self.gates.append(Gate("CZ", (c, t))); return self
    def swap(self, a, b): self.gates.append(Gate("SWAP", (a, b))); return self
    def measure(self, q): self.gates.append(Gate("MEASURE", (q,))); return self

    def is_clifford_circuit(self) -> bool:
        return all(g.is_clifford() or g.is_measurement() for g in self.gates)

    def depth_two_qubit(self) -> int:
        return sum(1 for g in self.gates if len(g.qubits) == 2)


def gate_unitary(name: str, params: Tuple[float, ...]) -> np.ndarray:
    if name == "H":
        return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    if name == "S":
        return np.array([[1, 0], [0, 1j]], dtype=complex)
    if name == "SDG":
        return np.array([[1, 0], [0, -1j]], dtype=complex)
    if name == "X":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if name == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    if name == "Z":
        return np.array([[1, 0], [0, -1]], dtype=complex)
    if name == "I":
        return np.eye(2, dtype=complex)
    if name == "T":
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    if name == "TDG":
        return np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)
    if name == "RX":
        th = params[0]
        c, s = np.cos(th / 2), np.sin(th / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if name == "RY":
        th = params[0]
        c, s = np.cos(th / 2), np.sin(th / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    if name == "RZ":
        th = params[0]
        return np.array([[np.exp(-1j * th / 2), 0], [0, np.exp(1j * th / 2)]], dtype=complex)
    raise ValueError(f"unknown 1q gate {name}")


def gate_unitary_2q(name: str) -> np.ndarray:
    if name == "CNOT":
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    if name == "CZ":
        return np.diag([1, 1, 1, -1]).astype(complex)
    if name == "SWAP":
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    raise ValueError(f"unknown 2q gate {name}")
