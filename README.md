WARNING: if you are a vibecoder or smth plz genuinly find a job 
also like yea 
coding is kool kid's work
make math for me 
matrix yeah 
i created it lol 
signed kds1123001 aka koolkid45 aka your boy skulltuber67 aka ezdomernuggetkid 
i dont do issues i do skibidies 
jessica is 2 kms away 
virus.bat loolllllll


#qsim — Stabilizer Tableau + MPS Quantum Circuit Simulator


----d o n t copy work lollolololololloolololololololololol 

Two simulation backends behind one gate/circuit IR, dispatched automatically:

- **`qsim/tableau.py`** — full Aaronson-Gottesman stabilizer tableau (CHP algorithm).
  Simulates arbitrary-width Clifford circuits (H, S, X, Y, Z, CNOT, CZ, SWAP,
  computational-basis measurement) in polynomial time via a `2n×(2n+1)` binary
  symplectic tableau. No exponential blowup — this is Gottesman-Knill, not a
  restricted state vector.
- **`qsim/mps.py`** — matrix product state engine for general (non-Clifford) circuits.
  Applies arbitrary 1- and 2-qubit gates via local SVD with bond-dimension
  truncation; non-adjacent qubits are handled via a swap network. Supports
  entanglement entropy at any cut, bitstring sampling, and Kraus-channel noise
  trajectories.
- **`qsim/hybrid_engine.py`** — `run_circuit()` inspects the circuit and routes
  pure-Clifford circuits to the tableau (exact, fast, scales to 1000+ qubits)
  and anything with T-gates/rotations/general unitaries to the MPS engine.

## What's actually verified (see `tests/`)

- `test_stabilizer_correctness.py` — tableau output checked against a brute-force
  dense state-vector simulator on Bell/GHZ states and 8 random Clifford circuits
  (n=5, depth 40), fidelity > 1 − 1e-6 in every case. Measurement statistics
  checked against the Born rule over 4000 trials.
- `test_mps_correctness.py` — MPS with `chi_max=1024` (no effective truncation)
  matches dense state-vector fidelity > 1 − 1e-8 on random circuits mixing H, S,
  X/Y/Z, T, RX/RY/RZ, CNOT, CZ, SWAP, including non-adjacent two-qubit gates
  routed through the swap network. Entanglement entropy verified exactly on
  Bell pairs (1 bit), GHZ chains (flat 1 bit at every cut), and product states
  (0 bits). Truncating `chi_max` down to 2 on a volume-law circuit is shown to
  measurably reduce fidelity relative to the untruncated run — i.e. the
  truncation error is real and the test catches it, not just asserted away.
- `test_bit_flip_code.py` — 3-qubit bit-flip repetition code: encode, inject a
  single X error on each of the 3 physical qubits, extract the ZZ syndrome via
  ancilla CNOTs, apply the correction, decode. Logical bit recovered correctly
  in all cases; syndrome is `(0,0)` for no error and nonzero for each injected
  error.
- `test_noise.py` — depolarizing Pauli-channel flip rate on the tableau and
  amplitude-damping decay rate via Kraus trajectories on the MPS engine both
  land within statistical tolerance of the analytic rate over thousands of
  trials.

Run everything: `python3 tests/test_*.py` (each file is also a standalone script).

## What's benchmarked, not just claimed

- `benchmarks/bench_stabilizer.py` — random Clifford circuits from n=10 to
  n=1000 qubits, depth 20. On this machine: ~0.02s at n=100, ~0.6s at n=1000.
  This qubit count is *entirely* out of reach for dense or MPS simulation
  (2^1000 states) — this is the actual payoff of Gottesman-Knill, demonstrated
  rather than asserted.
- `benchmarks/bench_mps.py` — a 24-qubit RY+CNOT brickwork circuit at
  `chi_max` from 2 to 64. Shows `max_bond` saturating at the cap and
  mid-chain entanglement entropy climbing toward `log2(chi_max)` as the cap is
  relaxed — i.e. the benchmark demonstrates the truncation is doing real work,
  not just running without error.

## Known limitations (unvalidated / intentionally out of scope)

- **MPS canonicalization is a full left-QR sweep after every 2-qubit gate**,
  not an incrementally maintained canonical form. This is O(n·χ³) per gate
  instead of the O(χ³) a production TEBD implementation would achieve — it is
  *correct* (verified above) but not the performance-competitive design a
  "high-performance" claim would imply for large n. This is the main honest
  gap relative to the brief's "high-performance" framing.
- **No long-range gate fusion / gate-set optimization, no GPU backend, no
  parallelism.** Single-threaded numpy throughout.
- **`to_statevector()` on either backend is exponential and hard-capped**
  (n≤14 for tableau reconstruction via stabilizer projectors, n≤20 for MPS
  contraction, n≤24 for the dense reference) — it exists for testing against
  ground truth, not as a scalable feature.
- **Noise support is real but narrow**: Pauli-twirl channels for the
  stabilizer engine (X/Y/Z with given probabilities — this keeps circuits
  inside the stabilizer formalism, so it cannot represent e.g. amplitude
  damping exactly on that backend) and single-qubit Kraus-operator quantum
  trajectories for the MPS engine (bit-flip, phase-flip, depolarizing,
  amplitude damping). No two-qubit correlated noise, no readout error model,
  no density-matrix (mixed-state) simulation on either backend — everything
  is pure-state trajectory sampling.
- **No true tensor-network contraction engine** in the sense of a general
  contraction-order optimizer (e.g. `cotengra`/`opt_einsum`-style) — the MPS
  is a 1D chain with a swap network for non-local gates, which is the right
  tool for lightly-entangled/NISQ-scale circuits but degrades for circuits
  that are genuinely 2D-connected (e.g. surface-code syndrome extraction on a
  grid) where a real tensor network with non-1D structure would do better.
- **Stabilizer→statevector reconstruction (`tableau.to_statevector`)** is
  implemented via explicit projector construction (`∏ (I + s_i)/2`) and
  eigendecomposition, which is correct but is the slowest, most brute-force
  way to do it — used only for n≤14 test/validation purposes, never in the
  benchmark path.

genuinly if you read all of ts please find a job man 
also gib me kandy and uhhh credit i guess keep opensource dont paste into slop machine
if i did the three example thing that Ai apparently does its bcuz like i got taught that you have to have 3 examples in your essays
i live by the rule of purrrfect grammer 
it should be eggcelent 
