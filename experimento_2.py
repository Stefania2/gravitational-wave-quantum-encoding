from __future__ import annotations

import cmath
import csv
import io
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2

BOLTZMANN_CONSTANT = 1.380649e-23
LIGHT_SPEED = 299_792_458
GRAVITATIONAL_CONSTANT = 6.67430e-11
REDUCED_PLANCK_CONSTANT = 1.054571817e-34

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
class SegmentResult:
    index: int
    gps_start: int
    gps_end: int
    n_qubits: int
    hilbert_dim: int
    mean_strain: float
    std_strain: float
    max_strain: float
    min_strain: float
    state_vector: list[complex] = field(repr=False)
    qft_spectrum: list[dict[str, float]]
    dominant_frequency: int
    cycle_period: int
    shannon_entropy: float
    spectral_concentration: float
    purity: float
    measured_state: str
    quantum_counts: dict[str, int] | None
    qiskit_used: bool
    elapsed_seconds: float


@dataclass
class ExperimentConfig:
    n_qubits: int = 8
    max_segments: int = 10
    sampling_rate: int = 4096
    use_qiskit_qft: bool = True
    shots: int = 4096
    optimization_level: int = 1


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def load_strain_data(hdf5_path: str) -> np.ndarray:
    import h5py
    print(f"Cargando datos de deformacion desde: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        strain = f['strain/Strain'][:]
    valid = strain[~np.isnan(strain)]
    print(f"  Muestras totales: {len(strain)}")
    print(f"  Muestras validas: {len(valid)}")
    return valid


def strain_segments(data: np.ndarray, n_qubits: int, max_segments: int, sampling_rate: int) -> list[tuple[int, int, np.ndarray]]:
    segment_size = 2 ** n_qubits
    segments: list[tuple[int, int, np.ndarray]] = []
    for i in range(min(max_segments, len(data) // segment_size)):
        start = i * segment_size
        end = start + segment_size
        segment = data[start:end]
        segments.append((start, end, segment))
    return segments


def normalize_state_vector(segment: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(segment)
    if norm == 0:
        return np.ones_like(segment) / math.sqrt(len(segment))
    return segment / norm


def planck_area() -> float:
    return GRAVITATIONAL_CONSTANT * REDUCED_PLANCK_CONSTANT / LIGHT_SPEED ** 3


def information_area_from_memory(memory_index: int) -> float:
    return 4 * max(1, memory_index) * planck_area()


def information_entropy(area: float) -> float:
    numerator = BOLTZMANN_CONSTANT * LIGHT_SPEED ** 3 * area
    denominator = 4 * GRAVITATIONAL_CONSTANT * REDUCED_PLANCK_CONSTANT
    return numerator / denominator


def shannon_entropy(probabilities: list[float]) -> float:
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


def spectral_concentration(spectrum: list[complex]) -> float:
    magnitudes = [abs(a) ** 2 for a in spectrum]
    total = sum(magnitudes)
    return max(magnitudes) / total if total > 0 else 0.0


def purity_from_state(psi: list[complex]) -> float:
    return sum(abs(a) ** 4 for a in psi)


def run_qft_circuit(state_vector: list[complex], shots: int = 4096) -> tuple[dict[str, int], bool]:
    n = len(state_vector)
    n_qubits = int(math.log2(n))
    if 2 ** n_qubits != n:
        return {}, False

    try:
        qc = QuantumCircuit(n_qubits)
        qc.initialize(state_vector, range(n_qubits))
        qc.append(QFT(n_qubits), range(n_qubits))
        qc.measure_all()

        aer_backend = AerSimulator()
        pm = generate_preset_pass_manager(optimization_level=1, backend=aer_backend)
        isa = pm.run(qc)
        sampler = SamplerV2(mode=aer_backend)
        job = sampler.run([isa], shots=shots)
        counts = job.result()[0].data.meas.get_counts()
        return counts, True
    except Exception as e:
        print(f"Error en circuito QFT: {e}", file=sys.stderr)
        return {}, False


def qft_manual(psi: list[complex]) -> list[complex]:
    n = len(psi)
    factor = 1 / math.sqrt(n)
    return [
        factor * sum(psi[j] * cmath.exp(2j * math.pi * j * k / n) for j in range(n))
        for k in range(n)
    ]


def detect_cycle_period(spectrum: list[complex]) -> tuple[int, int]:
    n = len(spectrum)
    magnitudes = [abs(a) for a in spectrum]
    if not any(magnitudes):
        return n, 0
    dominant = max(range(n), key=lambda i: magnitudes[i])
    period = n // (dominant or 1)
    return period, dominant


def decode_logical_state(value: int) -> str:
    return TEMPORAL_CODES.get(value % 16, "estado desconocido")


def analyze_segment(segment: np.ndarray, index: int, gps_start: int, config: ExperimentConfig) -> SegmentResult:
    t0 = time.perf_counter()
    n = len(segment)
    n_qubits = config.n_qubits

    norm_vec = normalize_state_vector(segment)
    state_vector: list[complex] = [complex(float(x), 0.0) for x in norm_vec]

    counts, qiskit_used = run_qft_circuit(state_vector, shots=config.shots)

    spectrum = qft_manual(state_vector)
    cycle_period, dominant_frequency = detect_cycle_period(spectrum)

    probs = [abs(a) ** 2 for a in state_vector]
    total_prob = sum(probs)
    if total_prob > 0:
        probs = [p / total_prob for p in probs]

    entropy = shannon_entropy(probs)
    spec_conc = spectral_concentration(spectrum)
    purity = purity_from_state(state_vector)

    measured_state = decode_logical_state(dominant_frequency)

    elapsed = time.perf_counter() - t0

    return SegmentResult(
        index=index,
        gps_start=gps_start,
        gps_end=gps_start + n,
        n_qubits=n_qubits,
        hilbert_dim=n,
        mean_strain=float(np.mean(segment)),
        std_strain=float(np.std(segment)),
        max_strain=float(np.max(segment)),
        min_strain=float(np.min(segment)),
        state_vector=state_vector,
        qft_spectrum=[{"frequency": k, "magnitude": abs(a)}
                       for k, a in enumerate(spectrum)],
        dominant_frequency=dominant_frequency,
        cycle_period=cycle_period,
        shannon_entropy=entropy,
        spectral_concentration=spec_conc,
        purity=purity,
        measured_state=measured_state,
        quantum_counts=counts,
        qiskit_used=qiskit_used,
        elapsed_seconds=elapsed,
    )


def result_to_text(r: SegmentResult) -> str:
    lines = []
    lines.append("-" * 60)
    lines.append(f"SEGMENTO {r.index}")
    lines.append("-" * 60)
    lines.append(f"  GPS start:      {r.gps_start}")
    lines.append(f"  GPS end:        {r.gps_end}")
    lines.append(f"  Qubits:         {r.n_qubits}")
    lines.append(f"  Dim. Hilbert:   {r.hilbert_dim}")
    lines.append(f"  Strain mean:    {r.mean_strain:.3e}")
    lines.append(f"  Strain std:     {r.std_strain:.3e}")
    lines.append(f"  Strain range:   [{r.min_strain:.3e}, {r.max_strain:.3e}]")
    lines.append("")
    lines.append(f"  Frec. dominante: {r.dominant_frequency}")
    lines.append(f"  Periodo ciclo:   {r.cycle_period}")
    lines.append(f"  Interpretacion:  {r.measured_state}")
    lines.append(f"  Entropia:        {r.shannon_entropy:.6f}")
    lines.append(f"  Conc. espectral: {r.spectral_concentration:.6f}")
    lines.append(f"  Pureza:          {r.purity:.6f}")
    lines.append(f"  Qiskit QFT:      {'si' if r.qiskit_used else 'no'}")
    lines.append(f"  Tiempo:          {r.elapsed_seconds:.3f}s")
    if r.quantum_counts:
        total_shots = sum(r.quantum_counts.values())
        lines.append("  Conteos QFT:")
        for state, count in sorted(r.quantum_counts.items()):
            pct = count / total_shots * 100
            lines.append(f"    |{state}>: {count} ({pct:.1f}%)")

    info_area = information_area_from_memory(r.cycle_period)
    info_ent = information_entropy(info_area)
    lines.append(f"  Area informacion: {info_area:.3e}")
    lines.append(f"  Entropia inform.: {info_ent:.3e}")
    return "\n".join(lines)


def results_to_csv(results: list[SegmentResult]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "segmento", "gps_start", "gps_end", "n_qubits", "dim_hilbert",
        "strain_mean", "strain_std", "strain_max", "strain_min",
        "frec_dominante", "periodo_ciclo", "interpretacion",
        "entropia_shannon", "concentracion_espectral", "pureza",
        "area_informacion", "entropia_informacion",
        "qiskit_usado", "tiempo_seg"
    ])
    for r in results:
        info_area = information_area_from_memory(r.cycle_period)
        info_ent = information_entropy(info_area)
        writer.writerow([
            r.index, r.gps_start, r.gps_end, r.n_qubits, r.hilbert_dim,
            f"{r.mean_strain:.6e}", f"{r.std_strain:.6e}",
            f"{r.max_strain:.6e}", f"{r.min_strain:.6e}",
            r.dominant_frequency, r.cycle_period, r.measured_state,
            f"{r.shannon_entropy:.6f}", f"{r.spectral_concentration:.6f}",
            f"{r.purity:.6f}", f"{info_area:.6e}", f"{info_ent:.6e}",
            "si" if r.qiskit_used else "no", f"{r.elapsed_seconds:.3f}",
        ])

    writer.writerow([])
    writer.writerow(["segmento", "frecuencia", "magnitud_qft"])
    for r in results:
        for item in r.qft_spectrum:
            writer.writerow([r.index, item["frequency"], f"{item['magnitude']:.6e}"])

    if results and results[0].quantum_counts:
        writer.writerow([])
        writer.writerow(["segmento", "estado_cuantico", "conteo"])
        for r in results:
            if r.quantum_counts:
                for state, count in r.quantum_counts.items():
                    writer.writerow([r.index, f"|{state}>", count])

    return output.getvalue()


def run_experiment(config: ExperimentConfig, hdf5_path: str, binary_path: str | None = None):
    print("=" * 60)
    print("EXPERIMENTO 2: ONDAS GRAVITACIONALES → ESTADOS CUANTICOS")
    print("=" * 60)
    print(f"Config: n_qubits={config.n_qubits}, segmentos_max={config.max_segments}, shots={config.shots}")
    print()

    strain = load_strain_data(hdf5_path)

    gps_start = 1421381632
    segments = strain_segments(strain, config.n_qubits, config.max_segments, config.sampling_rate)
    print(f"Procesando {len(segments)} segmentos de {2**config.n_qubits} muestras cada uno\n")

    results: list[SegmentResult] = []
    for i, (start_idx, end_idx, segment) in enumerate(segments):
        seg_gps = gps_start + start_idx // config.sampling_rate
        r = analyze_segment(segment, i, seg_gps, config)
        results.append(r)
        print(result_to_text(r))
        print()

    gps_end = gps_start + len(strain) // config.sampling_rate
    print("=" * 60)
    print("RESUMEN GLOBAL")
    print("=" * 60)
    print(f"Rango GPS: {gps_start} - {gps_end}")
    print(f"Total segmentos: {len(results)}")
    print(f"Entropia media: {np.mean([r.shannon_entropy for r in results]):.6f}")
    print(f"Pureza media: {np.mean([r.purity for r in results]):.6f}")
    print(f"Conc. espectral media: {np.mean([r.spectral_concentration for r in results]):.6f}")

    csv_data = results_to_csv(results)
    csv_filename = "estados_temporales_cuanticos.csv"
    with open(csv_filename, "w") as f:
        f.write(csv_data)
    print(f"\nResultados exportados a: {csv_filename}")

    if binary_path and os.path.exists(binary_path):
        print(f"\nProcesando archivo binario: {binary_path}")
        _process_binary_file(binary_path, config)


def _process_binary_file(path: str, config: ExperimentConfig):
    import hashlib
    with open(path, "rb") as f:
        raw = f.read()
    print(f"  Tamanio: {len(raw)} bytes")
    print(f"  SHA256: {hashlib.sha256(raw).hexdigest()}")

    bits = ''.join(format(b, '08b') for b in raw)
    print(f"  Bits totales: {len(bits)}")

    n = 2 ** config.n_qubits
    if len(bits) >= n:
        sample_bits = bits[:n]
        vals = np.array([int(b) for b in sample_bits], dtype=float)
        vals = vals * 2 - 1
        norm_vec = normalize_state_vector(vals)
        state_vector = [complex(float(x), 0.0) for x in norm_vec]
        spectrum = qft_manual(state_vector)
        period, dominant = detect_cycle_period(spectrum)
        entropy = shannon_entropy([abs(a) ** 2 for a in state_vector])
        spec_conc = spectral_concentration(spectrum)
        measured = decode_logical_state(dominant)

        print(f"  Vector estado ({config.n_qubits} qubits, dim={n})")
        print(f"  Frec. dominante: {dominant}")
        print(f"  Periodo: {period}")
        print(f"  Interpretacion: {measured}")
        print(f"  Entropia: {entropy:.6f}")
        print(f"  Conc. espectral: {spec_conc:.6f}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    config = ExperimentConfig(
        n_qubits=8,
        max_segments=10,
        sampling_rate=4096,
        use_qiskit_qft=True,
        shots=4096,
        optimization_level=1,
    )

    hdf5_path = os.path.join(base_dir, "H-H1_GWOSC_O4b_4KHZ_R1-1421381632-4096.hdf5")
    binary_path = os.path.join(base_dir, "data_14365094_2026-06-23T23-20-39_g0")

    if not os.path.exists(hdf5_path):
        gwf_path = os.path.join(base_dir, "H-H1_GWOSC_O4b_4KHZ_R1-1421381632-4096.gwf")
        if os.path.exists(gwf_path):
            print("Usando archivo GWF en lugar de HDF5")
            hdf5_path = gwf_path
        else:
            print("ERROR: No se encontro archivo de datos de ondas gravitacionales")
            sys.exit(1)

    run_experiment(config, hdf5_path, binary_path)


if __name__ == "__main__":
    main()
