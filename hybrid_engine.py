from typing import List, Optional, Union
import numpy as np
from .circuit import Circuit
from .tableau import StabilizerTableau
from .mps import MPS
#i just descovered that caine from tadc is like eniac but like reverse 

def run_circuit(circuit: Circuit, chi_max: int = 64, svd_cutoff: float = 1e-10,
                 rng: Optional[np.random.Generator] = None) -> Union[StabilizerTableau, MPS]:
    if circuit.is_clifford_circuit():
        engine = StabilizerTableau(circuit.n_qubits, rng=rng)
    else:
        engine = MPS(circuit.n_qubits, chi_max=chi_max, svd_cutoff=svd_cutoff, rng=rng)
    engine.run(circuit)
    return engine
