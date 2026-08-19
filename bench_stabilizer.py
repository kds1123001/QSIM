import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qsim.tableau import StabilizerTableau
from qsim.circuit import Gate
#as you guys know this is kind of my first sewious quantum computing project lolcat

CLIFFORD_1Q = ["H", "S", "X", "Y", "Z"]


def random_clifford_layer(n, rng):
    ops = []
    for q in range(n):
        ops.append((CLIFFORD_1Q[int(rng.integers(0, len(CLIFFORD_1Q)))], q))
    perm = rng.permutation(n)
    for i in range(0, n - 1, 2):
        ops.append(("CNOT", int(perm[i]), int(perm[i + 1])))
    return ops


def bench(n, depth, seed=0):
    rng = np.random.default_rng(seed)
    tab = StabilizerTableau(n, rng=rng)
    t0 = time.perf_counter()
    for _ in range(depth):
        for op in random_clifford_layer(n, rng):
            if op[0] == "CNOT":
                tab.cnot(op[1], op[2])
            elif op[0] == "H":
                tab.h(op[1])
            elif op[0] == "S":
                tab.s(op[1])
            elif op[0] == "X":
                tab.x_gate(op[1])
            elif op[0] == "Y":
                tab.y_gate(op[1])
            elif op[0] == "Z":
                tab.z_gate(op[1])
    elapsed = time.perf_counter() - t0
    return elapsed


if __name__ == "__main__":
    print(f"{'n_qubits':>10} {'depth':>8} {'seconds':>10} {'gates/sec':>12}")
    for n in [10, 50, 100, 300, 600, 1000]:
        depth = 20
        t = bench(n, depth)
        gates = depth * (n + n // 2)
        print(f"{n:>10} {depth:>8} {t:>10.4f} {gates/t:>12.0f}")
    print()
    print("This demonstrates the Gottesman-Knill polynomial-time (empirically ~O(n^2) per gate")
    print("via dense bit tableau row ops) scaling of the stabilizer simulator to qubit counts")
    print("that are entirely inaccessible to dense state-vector simulation (2^1000 is intractable).")
