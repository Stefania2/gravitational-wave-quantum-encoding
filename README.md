# Gravitational Wave Quantum Encoding

Quantum encoding of LIGO O4b gravitational wave strain data using Qiskit. Compares **amplitude encoding** and **angle encoding** strategies, demonstrating that amplitude encoding of GW strain collapses to the maximally mixed state while angle encoding preserves structure.

## Files

| File | Description |
|------|-------------|
| `experimento_2.py` | Amplitude encoding (8 qubits, QFT, measurement) |
| `experimento_angle_encoding.py` | Angle encoding via RY rotations |
| `experimento_hilbert_pequeno.py` | Hilbert dimension sweep (2–7 qubits) |
| `simulacion_cuantica.py` | Time-evolution simulation under diagonal Hamiltonian |
| `paper_ieee.tex` | LaTeX paper in IEEE format |
| `paper_ieee.html` | HTML version of the paper |

## Key Results

- **Purity–kurtosis identity**: $P = \kappa_{\text{eff}} / N$ (exact)
- **Entropy deviation**: $\Delta S = (NP - 1)/(2\ln 2) + \beta(NP - 1)^2$
- Amplitude encoding produces near-uniform states (entropy $\geq 94.6\%$ of max)
- Angle encoding yields structured states (entropy $\approx 3.68$ vs $8$ max)

## Requirements

- Python 3.12+
- Qiskit 2.4.2, Qiskit-Aer 0.17.2
- NumPy, h5py, matplotlib

## Data

LIGO O4b H1 strain channel (GPS 1421381632–1421385728, 4 kHz). Available at [GWOSC](https://gwosc.org/).

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC--BY--NC%204.0-green.svg)](https://creativecommons.org/licenses/by-nc/4.0/)


- Software: MIT License
- Documentation and figures: CC BY-NC 4.0
