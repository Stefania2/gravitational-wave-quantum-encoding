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
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2

TEMPORAL_CODES = {
    0: "estado estable", 1: "eco del pasado", 2: "recuerdo recurrente",
    3: "bifurcacion logica", 4: "linea alternativa", 5: "salto de indice",
    6: "paradoja estable", 7: "convergencia de ciclos", 8: "memoria persistente",
    9: "evento repetido", 10: "interferencia de fase", 11: "reconstruccion simbolica",
    12: "rama secundaria", 13: "frontera de informacion", 14: "sincronizacion de fases",
    15: "recurrencia completa",
}


@dataclass
class DimResult:
    n_qubits: int
    hilbert_dim: int
    segment_index: int
    strain_values: list[float]
    state_vector: list[complex]
    qft_spectrum: list[dict[str, float]]
    dominant_frequency: int
    shannon_entropy: float
    purity: float
    spectral_concentration: float
    measured_state: str
    variance_ratio: float
    max_min_ratio: float
    quantum_counts: dict[str, int] | None


@dataclass
class SummaryRow:
    n_qubits: int
    hilbert_dim: int
    avg_entropy: float
    avg_purity: float
    avg_spec_conc: float
    std_entropy: float
    std_purity: float
    std_spec_conc: float
    unique_states: int
    dominant_freqs: list[int]


def load_strain(hdf5_path: str) -> np.ndarray:
    import h5py
    with h5py.File(hdf5_path, 'r') as f:
        s = f['strain/Strain'][:]
    return s[~np.isnan(s)]


def normalize(seg: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(seg)
    return seg / n if n > 0 else np.ones_like(seg) / math.sqrt(len(seg))


def qft_manual(psi: list[complex]) -> list[complex]:
    n = len(psi)
    f = 1 / math.sqrt(n)
    return [f * sum(psi[j] * cmath.exp(2j * math.pi * j * k / n) for j in range(n)) for k in range(n)]


def purity(psi: list[complex]) -> float:
    return sum(abs(a) ** 4 for a in psi)


def spec_conc(spectrum: list[complex]) -> float:
    m = [abs(a) ** 2 for a in spectrum]
    t = sum(m)
    return max(m) / t if t > 0 else 0.0


def shannon(p: list[float]) -> float:
    return -sum(x * math.log2(x) for x in p if x > 0)


def decode(v: int) -> str:
    return TEMPORAL_CODES.get(v % 16, "?")


def run_qc(state: list[complex], shots: int = 4096) -> dict[str, int]:
    n = len(state)
    nq = int(math.log2(n))
    if 2 ** nq != n:
        return {}
    try:
        qc = QuantumCircuit(nq)
        qc.initialize(state, range(nq))
        qc.measure_all()
        backend = AerSimulator()
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        sampler = SamplerV2(mode=backend)
        job = sampler.run([pm.run(qc)], shots=shots)
        return job.result()[0].data.meas.get_counts()
    except Exception:
        return {}


def analyze_dim(segment: np.ndarray, n_qubits: int, idx: int) -> DimResult:
    seg_size = 2 ** n_qubits
    if len(segment) < seg_size:
        seg = np.pad(segment, (0, seg_size - len(segment)), 'edge')[:seg_size]
    else:
        seg = segment[:seg_size]

    normed = normalize(seg)
    psi = [complex(float(x), 0.0) for x in normed]
    spectrum = qft_manual(psi)
    mags = [abs(a) for a in spectrum]
    dom = max(range(len(mags)), key=lambda i: mags[i]) if any(mags) else 0
    probs = [abs(v) ** 2 for v in psi]

    var_ratio = float(np.var(seg) / np.var(segment[:seg_size])) if np.var(segment[:seg_size]) > 0 else 0.0
    max_min = float((seg.max() - seg.min()) / (segment[:seg_size].max() - segment[:seg_size].min())) if (segment[:seg_size].max() - segment[:seg_size].min()) > 1e-40 else 1.0

    counts = run_qc(psi)
    return DimResult(
        n_qubits=n_qubits, hilbert_dim=seg_size, segment_index=idx,
        strain_values=[float(x) for x in seg],
        state_vector=psi, qft_spectrum=[{"f": k, "m": abs(a)} for k, a in enumerate(spectrum)],
        dominant_frequency=dom,
        shannon_entropy=shannon(probs),
        purity=purity(psi),
        spectral_concentration=spec_conc(spectrum),
        measured_state=decode(dom),
        variance_ratio=var_ratio, max_min_ratio=max_min,
        quantum_counts=counts,
    )


def generate_plots(all_results: dict[int, list[DimResult]], base_dir: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    qubits_list = sorted(all_results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    colors = {2: 'red', 3: 'orange', 4: 'green', 5: 'blue', 6: 'purple', 7: 'brown', 8: 'gray'}

    # 1. Entropy vs Hilbert dimension (log scale)
    ax = axes[0, 0]
    for nq in qubits_list:
        ress = all_results[nq]
        hd = ress[0].hilbert_dim
        ents = [r.shannon_entropy for r in ress]
        ax.scatter([hd] * len(ents), ents, color=colors.get(nq, 'black'), alpha=0.5, s=30)
        ax.plot(hd, np.mean(ents), marker='D', color=colors.get(nq, 'black'), markersize=8, label=f'{nq}q (dim={hd})')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Dimension Hilbert (log2)')
    ax.set_ylabel('Entropia Shannon')
    ax.set_title('Entropia vs Dimension')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 2. Purity vs Hilbert dimension
    ax = axes[0, 1]
    for nq in qubits_list:
        ress = all_results[nq]
        hd = ress[0].hilbert_dim
        purs = [r.purity for r in ress]
        ax.scatter([hd] * len(purs), purs, color=colors.get(nq, 'black'), alpha=0.5, s=30)
        ax.plot(hd, np.mean(purs), marker='D', color=colors.get(nq, 'black'), markersize=8)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Dimension Hilbert (log2)')
    ax.set_ylabel('Pureza')
    ax.set_title('Pureza vs Dimension')
    ax.grid(alpha=0.3)

    # 3. Spectral concentration vs Hilbert dimension
    ax = axes[0, 2]
    for nq in qubits_list:
        ress = all_results[nq]
        hd = ress[0].hilbert_dim
        scs = [r.spectral_concentration for r in ress]
        ax.scatter([hd] * len(scs), scs, color=colors.get(nq, 'black'), alpha=0.5, s=30)
        ax.plot(hd, np.mean(scs), marker='D', color=colors.get(nq, 'black'), markersize=8)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Dimension Hilbert (log2)')
    ax.set_ylabel('Conc. Espectral')
    ax.set_title('Concentracion Espectral vs Dimension')
    ax.grid(alpha=0.3)

    # 4. Entropy distribution per qubit count (boxplot-like)
    ax = axes[1, 0]
    labels = [f'{nq}q\ndim={all_results[nq][0].hilbert_dim}' for nq in qubits_list]
    data_ents = [[r.shannon_entropy for r in all_results[nq]] for nq in qubits_list]
    bp = ax.boxplot(data_ents, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], [colors[nq] for nq in qubits_list]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    max_ent = [math.log2(all_results[nq][0].hilbert_dim) for nq in qubits_list]
    ax.plot(range(1, len(qubits_list) + 1), max_ent, 'k--', label='Max teorica', alpha=0.5)
    ax.set_ylabel('Entropia Shannon')
    ax.set_title('Distribucion Entropia por Dimension')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 5. Purity distribution
    ax = axes[1, 1]
    data_pur = [[r.purity for r in all_results[nq]] for nq in qubits_list]
    bp = ax.boxplot(data_pur, tick_labels=labels, patch_artist=True)
    uniform_line = [1.0 / all_results[nq][0].hilbert_dim for nq in qubits_list]
    for patch, color in zip(bp['boxes'], [colors[nq] for nq in qubits_list]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.plot(range(1, len(qubits_list) + 1), uniform_line, 'k--', label='Uniforme (1/dim)', alpha=0.5)
    ax.set_ylabel('Pureza')
    ax.set_title('Distribucion Pureza por Dimension')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 6. Ratio varianza vs dimension (amplificacion)
    ax = axes[1, 2]
    for nq in qubits_list:
        ress = all_results[nq]
        hd = ress[0].hilbert_dim
        ratios = [r.variance_ratio for r in ress]
        ax.scatter([hd] * len(ratios), ratios, color=colors.get(nq, 'black'), alpha=0.5, s=30)
        ax.plot(hd, np.mean(ratios), marker='D', color=colors.get(nq, 'black'), markersize=8)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Dimension Hilbert (log2)')
    ax.set_ylabel('Razon varianza (submuestra/original)')
    ax.set_title('Amplificacion de diferencias relativas')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f'{base_dir}/comparacion_dimensiones_hilbert.png', dpi=150)
    plt.close(fig)
    print(f"  Grafico: {base_dir}/comparacion_dimensiones_hilbert.png")

    # Frecuencia dominante por dimension
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(qubits_list))
    width = 0.8 / max(1, max(len(all_results[nq]) for nq in qubits_list))
    for i, nq in enumerate(qubits_list):
        ress = all_results[nq]
        doms = [r.dominant_frequency for r in ress]
        for j, d in enumerate(doms):
            ax.bar(i + j * width - 0.4, d, width, color=colors.get(nq, 'black'), alpha=0.6 + 0.4 * j / max(1, len(doms)))
    ax.set_xticks(range(len(qubits_list)))
    ax.set_xticklabels([f'{nq}q (dim={all_results[nq][0].hilbert_dim})' for nq in qubits_list])
    ax.set_ylabel('Frecuencia Dominante')
    ax.set_title('Frecuencia Dominante por Dimension Hilbert (cada barra = 1 segmento)')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(f'{base_dir}/frecuencia_dominante_por_dim.png', dpi=150)
    plt.close(fig)
    print(f"  Grafico: {base_dir}/frecuencia_dominante_por_dim.png")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    hdf5_path = os.path.join(base_dir, "H-H1_GWOSC_O4b_4KHZ_R1-1421381632-4096.hdf5")
    if not os.path.exists(hdf5_path):
        print("ERROR: No se encuentra archivo HDF5")
        sys.exit(1)

    # Probamos de 2 a 7 qubits (Hilbert dims: 4, 8, 16, 32, 64, 128)
    qubit_range = [2, 3, 4, 5, 6, 7]
    n_segments = 10

    strain = load_strain(hdf5_path)
    print("=" * 70)
    print("EXPERIMENTO: ESPACIOS DE HILBERT PEQUENOS")
    print("Amplificacion de diferencias relativas en datos GW")
    print("=" * 70)
    print(f"  Segmentos por configuracion: {n_segments}")
    print(f"  Rango de qubits: {qubit_range}")
    print(f"  Dimensiones Hilbert: {[2**q for q in qubit_range]}")
    print()

    all_results: dict[int, list[DimResult]] = {}

    # Para cada configuracion de qubits, usamos los mismos segmentos base
    # pero variando cuantos datos entran en cada estado
    for nq in qubit_range:
        print(f"{'-'*70}")
        print(f"  QUBITS={nq}  |  DIM HILBERT={2**nq}")
        print(f"{'-'*70}")

        seg_size = 2 ** nq
        total_needed = seg_size * n_segments
        if total_needed > len(strain):
            actual_seg = len(strain) // seg_size
            n_use = min(n_segments, actual_seg)
        else:
            n_use = n_segments

        results: list[DimResult] = []
        for i in range(n_use):
            seg = strain[i * seg_size:(i + 1) * seg_size]
            r = analyze_dim(seg, nq, i)

            probs_str = " ".join(
                f"|{format(j, f'0{nq}b')}>:{abs(r.state_vector[j])**2:.3f}"
                for j in range(min(4, r.hilbert_dim))
            )
            print(f"    Seg {i}: entrop={r.shannon_entropy:.4f}  pureza={r.purity:.6f}  "
                  f"frecDom={r.dominant_frequency}  estado={r.measured_state}  "
                  f"varRatio={r.variance_ratio:.3f}  {probs_str}{'...' if r.hilbert_dim > 4 else ''}")

            results.append(r)

        avg_e = np.mean([r.shannon_entropy for r in results])
        avg_p = np.mean([r.purity for r in results])
        avg_sc = np.mean([r.spectral_concentration for r in results])
        print(f"    -> Promedio: entrop={avg_e:.4f}  pureza={avg_p:.6f}  concEsp={avg_sc:.6f}")
        print()

        all_results[nq] = results

    print("=" * 70)
    print("COMPARACION CRUZADA POR DIMENSION HILBERT")
    print("=" * 70)
    print(f"  {'Qubits':<7} {'Dim':<6} {'Entropia':<10} {'Pureza':<10} {'Conc.Esp':<10} {'VarRatio':<10}")
    print(f"  {'-'*7} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for nq in qubit_range:
        ress = all_results[nq]
        avg_e = np.mean([r.shannon_entropy for r in ress])
        avg_p = np.mean([r.purity for r in ress])
        avg_sc = np.mean([r.spectral_concentration for r in ress])
        avg_vr = np.mean([r.variance_ratio for r in ress])
        print(f"  {nq:<7} {2**nq:<6} {avg_e:<10.4f} {avg_p:<10.6f} {avg_sc:<10.6f} {avg_vr:<10.3f}")

    print()

    # Conclusion
    print("CONCLUSION:")
    prev_ent = None
    for nq in qubit_range:
        ress = all_results[nq]
        avg_e = np.mean([r.shannon_entropy for r in ress])
        max_e = math.log2(2 ** nq)
        ratio = avg_e / max_e if max_e > 0 else 0
        delta = f"  (vs ant: {prev_ent - avg_e:+.4f})" if prev_ent is not None else ""
        print(f"  {nq}q (dim={2**nq}): entrop={avg_e:.4f}/{max_e:.4f} ({ratio*100:.1f}% del maximo){delta}")
        prev_ent = avg_e

    generate_plots(all_results, base_dir)

    # Exportar CSV
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["n_qubits", "dim_hilbert", "segmento", "entropia", "pureza",
                 "conc_espectral", "frec_dominante", "estado", "varianza_ratio",
                 "max_min_ratio"])

    for nq in qubit_range:
        for r in all_results[nq]:
            w.writerow([r.n_qubits, r.hilbert_dim, r.segment_index,
                        f"{r.shannon_entropy:.6f}", f"{r.purity:.6f}",
                        f"{r.spectral_concentration:.6f}", r.dominant_frequency,
                        r.measured_state, f"{r.variance_ratio:.6f}",
                        f"{r.max_min_ratio:.6f}"])

    csv_path = os.path.join(base_dir, "comparacion_dimensiones_hilbert.csv")
    with open(csv_path, "w") as f:
        f.write(out.getvalue())
    print(f"\nResultados exportados a: {csv_path}")


if __name__ == "__main__":
    main()
