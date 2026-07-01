from __future__ import annotations

import cmath
import csv
import io
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2

TEMPORAL_CODES = {
    0: "estado estable",
    1: "eco del pasado",
    2: "recuerdo recurrente",
    3: "bifurcacion logica",
    4: "linea alternativa",
    5: "salto de indice",
    6: "paradoja estable",
    7: "convergencia de ciclos",
    8: "memoria persistente",
    9: "evento repetido",
    10: "interferencia de fase",
    11: "reconstruccion simbolica",
    12: "rama secundaria",
    13: "frontera de informacion",
    14: "sincronizacion de fases",
    15: "recurrencia completa",
}


@dataclass
class AngleResult:
    segment_index: int
    n_qubits: int
    angles: list[float]
    raw_strain: list[float]
    amplitudes: list[complex]
    qft_spectrum: list[dict[str, float]]
    dominant_frequency: int
    cycle_period: int
    shannon_entropy: float
    spectral_concentration: float
    purity: float
    measured_state: str
    counts_before_qft: dict[str, int] | None
    counts_after_qft: dict[str, int] | None
    elapsed: float


def load_strain_data(hdf5_path: str) -> np.ndarray:
    import h5py
    with h5py.File(hdf5_path, 'r') as f:
        strain = f['strain/Strain'][:]
    return strain[~np.isnan(strain)]


def strain_to_angles(segment: np.ndarray) -> list[float]:
    seg_min = segment.min()
    seg_max = segment.max()
    if abs(seg_max - seg_min) < 1e-40:
        return [0.0] * len(segment)
    scaled = (segment - seg_min) / (seg_max - seg_min)
    return [float(v) * math.pi for v in scaled]


def build_angle_circuit(angles: list[float], apply_qft: bool = True) -> QuantumCircuit:
    n = len(angles)
    qc = QuantumCircuit(n)

    for i, theta in enumerate(angles):
        qc.ry(theta, i)

    if apply_qft:
        qc.append(QFT(n), range(n))

    qc.measure_all()
    return qc


def run_circuit(qc: QuantumCircuit, shots: int = 4096) -> dict[str, int]:
    backend = AerSimulator()
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa = pm.run(qc)
    sampler = SamplerV2(mode=backend)
    job = sampler.run([isa], shots=shots)
    return job.result()[0].data.meas.get_counts()


def state_vector_from_angles(angles: list[float]) -> list[complex]:
    n = len(angles)
    dim = 2 ** n
    psi = [complex(1.0, 0.0)] * dim
    for i in range(dim):
        amp = complex(1.0, 0.0)
        for q in range(n):
            bit = (i >> (n - 1 - q)) & 1
            theta = angles[q]
            if bit == 0:
                amp *= complex(math.cos(theta / 2), 0)
            else:
                amp *= complex(math.sin(theta / 2), 0)
        psi[i] = amp
    norm = math.sqrt(sum(abs(a) ** 2 for a in psi))
    return [a / norm for a in psi] if norm > 0 else psi


def qft_manual(psi: list[complex]) -> list[complex]:
    n = len(psi)
    f = 1 / math.sqrt(n)
    return [
        f * sum(psi[j] * cmath.exp(2j * math.pi * j * k / n) for j in range(n))
        for k in range(n)
    ]


def purity(psi: list[complex]) -> float:
    return sum(abs(a) ** 4 for a in psi)


def spectral_concentration(spectrum: list[complex]) -> float:
    mags = [abs(a) ** 2 for a in spectrum]
    t = sum(mags)
    return max(mags) / t if t > 0 else 0.0


def shannon_entropy(probs: list[float]) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)


def decode_state(val: int) -> str:
    return TEMPORAL_CODES.get(val % 16, "desconocido")


def analyze_angle_segment(segment: np.ndarray, idx: int) -> AngleResult:
    t0 = time.perf_counter()
    n_qubits = len(segment)

    angles = strain_to_angles(segment)
    raw = [float(v) for v in segment]

    psi = state_vector_from_angles(angles)
    spectrum = qft_manual(psi)

    mags = [abs(a) for a in spectrum]
    if any(mags):
        dom = max(range(len(mags)), key=lambda i: mags[i])
    else:
        dom = 0
    period = len(mags) // (dom or 1)

    probs = [abs(v) ** 2 for v in psi]
    tp = sum(probs)
    if tp > 0:
        probs = [p / tp for p in probs]

    qc_before = build_angle_circuit(angles, apply_qft=False)
    qc_after = build_angle_circuit(angles, apply_qft=True)

    counts_before = run_circuit(qc_before)
    counts_after = run_circuit(qc_after)

    elapsed = time.perf_counter() - t0

    return AngleResult(
        segment_index=idx,
        n_qubits=n_qubits,
        angles=angles,
        raw_strain=raw,
        amplitudes=psi,
        qft_spectrum=[
            {"frequency": k, "magnitude": abs(a)}
            for k, a in enumerate(spectrum)
        ],
        dominant_frequency=dom,
        cycle_period=period,
        shannon_entropy=shannon_entropy(probs),
        spectral_concentration=spectral_concentration(spectrum),
        purity=purity(psi),
        measured_state=decode_state(dom),
        counts_before_qft=counts_before,
        counts_after_qft=counts_after,
        elapsed=elapsed,
    )


def format_result(r: AngleResult) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"ANGLE ENCODING - SEGMENTO {r.segment_index}")
    lines.append("=" * 70)
    lines.append(f"  Qubits:         {r.n_qubits}")
    lines.append(f"  Angulos (rad):  {[f'{a:.4f}' for a in r.angles[:8]]}{'...' if len(r.angles) > 8 else ''}")
    lines.append(f"  Strain raw:     {[f'{s:.3e}' for s in r.raw_strain[:4]]}{'...' if len(r.raw_strain) > 4 else ''}")
    lines.append(f"  Frec dominante: {r.dominant_frequency}")
    lines.append(f"  Periodo:        {r.cycle_period}")
    lines.append(f"  Interpretacion: {r.measured_state}")
    lines.append(f"  Entropia:       {r.shannon_entropy:.6f}")
    lines.append(f"  Pureza:         {r.purity:.6f}")
    lines.append(f"  Conc. espectral:{r.spectral_concentration:.6f}")
    lines.append(f"  Tiempo:         {r.elapsed:.3f}s")

    if r.counts_before_qft:
        total = sum(r.counts_before_qft.values())
        top = sorted(r.counts_before_qft.items(), key=lambda x: -x[1])[:5]
        lines.append("  Top-5 conteos (sin QFT):")
        for state, c in top:
            lines.append(f"    |{state}>: {c} ({c/total*100:.1f}%)")

    if r.counts_after_qft:
        total = sum(r.counts_after_qft.values())
        top = sorted(r.counts_after_qft.items(), key=lambda x: -x[1])[:5]
        lines.append("  Top-5 conteos (con QFT):")
        for state, c in top:
            lines.append(f"    |{state}>: {c} ({c/total*100:.1f}%)")

    return "\n".join(lines)


def results_to_csv(results: list[AngleResult]) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow([
        "segmento", "n_qubits",
        "strain_min", "strain_max", "strain_mean", "strain_std",
        "frec_dominante", "periodo", "interpretacion",
        "entropia", "pureza", "conc_espectral", "tiempo_seg"
    ])
    for r in results:
        arr = np.array(r.raw_strain)
        w.writerow([
            r.segment_index, r.n_qubits,
            f"{arr.min():.6e}", f"{arr.max():.6e}",
            f"{arr.mean():.6e}", f"{arr.std():.6e}",
            r.dominant_frequency, r.cycle_period, r.measured_state,
            f"{r.shannon_entropy:.6f}", f"{r.purity:.6f}",
            f"{r.spectral_concentration:.6f}", f"{r.elapsed:.3f}",
        ])

    w.writerow([])
    w.writerow(["segmento", "frecuencia", "magnitud_qft"])
    for r in results:
        for item in r.qft_spectrum:
            w.writerow([r.segment_index, item["frequency"], f"{item['magnitude']:.6e}"])

    return out.getvalue()


def generate_plots(results: list[AngleResult], base_dir: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(results)
    indices = [r.segment_index for r in results]
    entropies = [r.shannon_entropy for r in results]
    purities = [r.purity for r in results]
    concs = [r.spectral_concentration for r in results]
    doms = [r.dominant_frequency for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].bar(indices, entropies, color='steelblue', alpha=0.7)
    axes[0, 0].set_xlabel('Segmento')
    axes[0, 0].set_ylabel('Entropia Shannon')
    axes[0, 0].set_title('Angle Encoding - Entropia por Segmento')
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].bar(indices, purities, color='coral', alpha=0.7)
    axes[0, 1].axhline(y=1/256, color='gray', linestyle='--', alpha=0.5, label='Uniforme (1/256)')
    axes[0, 1].set_xlabel('Segmento')
    axes[0, 1].set_ylabel('Pureza')
    axes[0, 1].set_title('Angle Encoding - Pureza por Segmento')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].bar(indices, concs, color='seagreen', alpha=0.7)
    axes[1, 0].set_xlabel('Segmento')
    axes[1, 0].set_ylabel('Concentracion Espectral')
    axes[1, 0].set_title('Angle Encoding - Conc. Espectral por Segmento')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].bar(indices, doms, color='mediumpurple', alpha=0.7)
    axes[1, 1].set_xlabel('Segmento')
    axes[1, 1].set_ylabel('Frecuencia Dominante')
    axes[1, 1].set_title('Angle Encoding - Frecuencia Dominante')
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f'{base_dir}/angle_encoding_summary.png', dpi=150)
    plt.close(fig)
    print(f"  Grafico: {base_dir}/angle_encoding_summary.png")

    # Espectro QFT comparativo
    n_plots = min(4, n)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))
    if n_plots == 1:
        axes = [axes]
    for i in range(n_plots):
        r = results[i]
        freqs = [item['frequency'] for item in r.qft_spectrum]
        mags = [item['magnitude'] for item in r.qft_spectrum]
        axes[i].bar(freqs, mags, width=1.0, color='steelblue', alpha=0.6)
        axes[i].set_xlabel('Frecuencia')
        axes[i].set_ylabel('Magnitud QFT')
        axes[i].set_title(f'Segmento {r.segment_index} - Espectro QFT (Angle Encoding)')
        axes[i].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{base_dir}/angle_qft_spectra.png', dpi=150)
    plt.close(fig)
    print(f"  Grafico: {base_dir}/angle_qft_spectra.png")

    # Comparacion: amplitude vs angle encoding
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in results:
        freqs = [item['frequency'] for item in r.qft_spectrum]
        mags = [item['magnitude'] for item in r.qft_spectrum]
        ax.plot(freqs, mags, label=f'S{r.segment_index}', linewidth=0.8)
    ax.set_xlabel('Frecuencia')
    ax.set_ylabel('Magnitud QFT')
    ax.set_title('Espectro QFT - Angle Encoding (todos los segmentos)')
    ax.legend(loc='best', ncol=2, fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{base_dir}/angle_qft_all_segments.png', dpi=150)
    plt.close(fig)
    print(f"  Grafico: {base_dir}/angle_qft_all_segments.png")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    hdf5_path = os.path.join(base_dir, "H-H1_GWOSC_O4b_4KHZ_R1-1421381632-4096.hdf5")
    if not os.path.exists(hdf5_path):
        print("ERROR: No se encuentra el archivo HDF5")
        sys.exit(1)

    n_qubits = 8
    max_segments = 10
    shots = 4096

    print("=" * 70)
    print("ANGLE ENCODING: STRAIN GW COMO ANGULOS DE ROTACION")
    print("=" * 70)
    print(f"  Qubits por segmento: {n_qubits}")
    print(f"  Segmentos a procesar: {max_segments}")
    print(f"  Shots por circuito: {shots}")
    print(f"  Total qubits usados: {n_qubits * max_segments}")
    print()

    strain = load_strain_data(hdf5_path)
    seg_size = n_qubits
    segments: list[np.ndarray] = []
    for i in range(min(max_segments, len(strain) // seg_size)):
        segments.append(strain[i * seg_size:(i + 1) * seg_size])

    print(f"Procesando {len(segments)} segmentos de {seg_size} muestras cada uno\n")

    results: list[AngleResult] = []
    for i, seg in enumerate(segments):
        print(f"--- Segmento {i} ---")
        r = analyze_angle_segment(seg, i)
        results.append(r)
        print(format_result(r))
        print()

    print("=" * 70)
    print("RESUMEN COMPARATIVO: ANGLE vs AMPLITUDE ENCODING")
    print("=" * 70)
    avg_ent = np.mean([r.shannon_entropy for r in results])
    avg_pur = np.mean([r.purity for r in results])
    avg_conc = np.mean([r.spectral_concentration for r in results])
    print(f"  Angle Encoding - Entropia media:  {avg_ent:.6f}")
    print(f"  Angle Encoding - Pureza media:    {avg_pur:.6f}")
    print(f"  Angle Encoding - Conc. esp media: {avg_conc:.6f}")

    import matplotlib
    matplotlib.use('Agg')
    generate_plots(results, base_dir)

    csv_data = results_to_csv(results)
    csv_path = os.path.join(base_dir, "angle_encoding_resultados.csv")
    with open(csv_path, "w") as f:
        f.write(csv_data)
    print(f"\nResultados exportados a: {csv_path}")
    print("Graficos generados: angle_encoding_summary.png, angle_qft_spectra.png, angle_qft_all_segments.png")


if __name__ == "__main__":
    main()
