import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qsim.mps import MPS
from qsim.circuit import Circuit, gate_unitary, gate_unitary_2q


def build_brickwork_circuit(n, layers, seed):
    rng = np.random.default_rng(seed)
    c = Circuit(n)
    for layer in range(layers):
        for q in range(n):
            c.gates.append(__import__("qsim.circuit", fromlist=["Gate"]).Gate(
                "RY", (q,), (float(rng.uniform(0, np.pi)),)))
        for q in range(layer % 2, n - 1, 2):
            c.cnot(q, q + 1)
    return c


def bench_chi(n, layers, chi_max, seed=0):
    c = build_brickwork_circuit(n, layers, seed)
    mps = MPS(n, chi_max=chi_max, svd_cutoff=1e-12, rng=np.random.default_rng(seed))
    t0 = time.perf_counter()
    mps.run(c)
    elapsed = time.perf_counter() - t0
    max_bond = max(mps.bond_dims()) if mps.bond_dims() else 1
    ent = mps.entanglement_entropy(n // 2 - 1)
    return elapsed, max_bond, ent


if __name__ == "__main__":
    n, layers = 24, 12
    print(f"brickwork RY+CNOT circuit, n={n} qubits, {layers} layers")
    print(f"{'chi_max':>10} {'seconds':>10} {'max_bond':>10} {'mid_entropy':>12}")
    for chi in [2, 4, 8, 16, 32, 64]:
        t, mb, ent = bench_chi(n, layers, chi)
        print(f"{chi:>10} {t:>10.4f} {mb:>10} {ent:>12.4f}")
    print()
    print("max_bond saturating at chi_max shows the truncation is active (entanglement wants")
    print("to exceed the cap); mid_entropy approaching log2(chi_max) confirms volume-law growth")
    print("that MPS with bounded bond dimension cannot represent exactly, unlike the stabilizer")
    print("simulator, which is exact but restricted to Clifford gates only.")
