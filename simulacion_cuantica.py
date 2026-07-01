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
class TimeStepResult:
    step: int
    time_value: float
    fidelity: float
    shannon_entropy: float
    purity: float
    spectral_concentration: float
    dominant_frequency: int
    measured_state: str
    quantum_counts: dict[str, int] | None


@dataclass
class SegmentSimulation:
    segment_index: int
    n_qubits: int
    hilbert_dim: int
    mean_strain: float
    std_strain: float
    hamiltonian_energy_scale: float
    time_steps: list[TimeStepResult] = field(default_factory=list)


def load_strain_data(hdf5_path: str) -> np.ndarray:
    import h5py
    with h5py.File(hdf5_path, 'r') as f:
        strain = f['strain/Strain'][:]
    valid = strain[~np.isnan(strain)]
    return valid


def strain_segments(data: np.ndarray, n_qubits: int, max_segments: int) -> list[tuple[int, int, np.ndarray]]:
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


def build_hamiltonian_parameters(strain_seg: np.ndarray) -> tuple[float, np.ndarray]:
    mean_s = float(np.mean(strain_seg))
    std_s = float(np.std(strain_seg))
    energy_scale = abs(mean_s) + std_s
    if energy_scale < 1e-30:
        energy_scale = 1e-30
    local_energies = strain_seg - mean_s
    return energy_scale, local_energies


def shannon_entropy(probs: list[float]) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)


def purity_from_psi(psi: list[complex]) -> float:
    return sum(abs(a) ** 4 for a in psi)


def spectral_concentration_from_psi(psi: list[complex]) -> float:
    n = len(psi)
    spectrum = [
        1 / math.sqrt(n) * sum(psi[j] * cmath.exp(2j * math.pi * j * k / n) for j in range(n))
        for k in range(n)
    ]
    magnitudes = [abs(a) ** 2 for a in spectrum]
    total = sum(magnitudes)
    return max(magnitudes) / total if total > 0 else 0.0


def dominant_freq_from_psi(psi: list[complex]) -> int:
    n = len(psi)
    spectrum = [
        1 / math.sqrt(n) * sum(psi[j] * cmath.exp(2j * math.pi * j * k / n) for j in range(n))
        for k in range(n)
    ]
    magnitudes = [abs(a) for a in spectrum]
    if not any(magnitudes):
        return 0
    return max(range(n), key=lambda i: magnitudes[i])


def decode_logical_state(value: int) -> str:
    return TEMPORAL_CODES.get(value % 16, "estado desconocido")


def prob_shannon(population: dict[str, int]) -> float:
    total = sum(population.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in population.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def run_initial_state_circuit(state_vector: list[complex], shots: int = 4096) -> dict[str, int]:
    n = len(state_vector)
    nq = int(math.log2(n))
    if 2 ** nq != n:
        return {}
    try:
        qc = QuantumCircuit(nq)
        qc.initialize(state_vector, range(nq))
        qc.measure_all()
        backend = AerSimulator()
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        isa = pm.run(qc)
        sampler = SamplerV2(mode=backend)
        job = sampler.run([isa], shots=shots)
        return job.result()[0].data.meas.get_counts()
    except Exception as e:
        print(f"  Error en medicion inicial: {e}", file=sys.stderr)
        return {}


def run_evolution_circuit(state_vector: list[complex], time_step: float, energy_scale: float, local_energies: np.ndarray, shots: int = 4096) -> dict[str, int]:
    n = len(state_vector)
    nq = int(math.log2(n))
    if 2 ** nq != n:
        return {}

    evolved = state_vector.copy()
    for i in range(n):
        phase = -local_energies[i] * time_step / energy_scale
        evolved[i] = state_vector[i] * cmath.exp(1j * phase)

    norm = math.sqrt(sum(abs(a) ** 2 for a in evolved))
    if norm > 0:
        evolved = [a / norm for a in evolved]

    try:
        qc = QuantumCircuit(nq)
        qc.initialize(evolved, range(nq))
        qc.measure_all()
        backend = AerSimulator()
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        isa = pm.run(qc)
        sampler = SamplerV2(mode=backend)
        job = sampler.run([isa], shots=shots)
        return job.result()[0].data.meas.get_counts()
    except Exception as e:
        print(f"  Error en circuito evolucion: {e}", file=sys.stderr)
        return {}


def simulate_segment(segment: np.ndarray, idx: int, n_time_steps: int = 16, shots: int = 4096) -> SegmentSimulation:
    n_qubits = int(math.log2(len(segment)))
    hilbert_dim = len(segment)

    norm_vec = normalize_state_vector(segment)
    state_vector: list[complex] = [complex(float(x), 0.0) for x in norm_vec]

    energy_scale, local_energies = build_hamiltonian_parameters(segment)

    sim = SegmentSimulation(
        segment_index=idx,
        n_qubits=n_qubits,
        hilbert_dim=hilbert_dim,
        mean_strain=float(np.mean(segment)),
        std_strain=float(np.std(segment)),
        hamiltonian_energy_scale=energy_scale,
    )

    probs = [abs(a) ** 2 for a in state_vector]
    total_p = sum(probs)
    if total_p > 0:
        probs_n = [p / total_p for p in probs]
    else:
        probs_n = probs

    init_counts = run_initial_state_circuit(state_vector, shots=shots)
    t0 = TimeStepResult(
        step=0,
        time_value=0.0,
        fidelity=1.0,
        shannon_entropy=shannon_entropy(probs_n),
        purity=purity_from_psi(state_vector),
        spectral_concentration=spectral_concentration_from_psi(state_vector),
        dominant_frequency=dominant_freq_from_psi(state_vector),
        measured_state=decode_logical_state(dominant_freq_from_psi(state_vector)),
        quantum_counts=init_counts,
    )
    sim.time_steps.append(t0)

    dt = 0.5 / energy_scale if energy_scale > 1e-30 else 1.0
    current_state = state_vector.copy()

    for step in range(1, n_time_steps + 1):
        t_val = step * dt

        evolved = current_state.copy()
        for i in range(hilbert_dim):
            phase = -local_energies[i] * t_val / energy_scale
            evolved[i] = state_vector[i] * cmath.exp(1j * phase)

        norm_ev = math.sqrt(sum(abs(a) ** 2 for a in evolved))
        if norm_ev > 0:
            evolved = [a / norm_ev for a in evolved]

        probs_ev = [abs(a) ** 2 for a in evolved]
        tp_ev = sum(probs_ev)
        if tp_ev > 0:
            probs_ev_n = [p / tp_ev for p in probs_ev]
        else:
            probs_ev_n = probs_ev

        fid = abs(sum(
            state_vector[i].conjugate() * evolved[i]
            for i in range(hilbert_dim)
        ))

        dom_freq = dominant_freq_from_psi(evolved)
        counts = run_evolution_circuit(current_state, t_val, energy_scale, local_energies, shots=shots)

        tr = TimeStepResult(
            step=step,
            time_value=t_val,
            fidelity=fid,
            shannon_entropy=shannon_entropy(probs_ev_n),
            purity=purity_from_psi(evolved),
            spectral_concentration=spectral_concentration_from_psi(evolved),
            dominant_frequency=dom_freq,
            measured_state=decode_logical_state(dom_freq),
            quantum_counts=counts,
        )
        sim.time_steps.append(tr)
        current_state = evolved

    return sim


def sim_to_text(sim: SegmentSimulation) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"SIMULACION SEGMENTO {sim.segment_index}")
    lines.append("=" * 70)
    lines.append(f"  Qubits:       {sim.n_qubits}")
    lines.append(f"  Dim. Hilbert:  {sim.hilbert_dim}")
    lines.append(f"  Strain mean:   {sim.mean_strain:.3e}")
    lines.append(f"  Strain std:    {sim.std_strain:.3e}")
    lines.append(f"  Escala energia: {sim.hamiltonian_energy_scale:.3e}")
    lines.append("")
    lines.append(f"  {'Paso':<5} {'Tiempo':<12} {'Fidelidad':<12} {'Entropia':<10} {'Pureza':<10} {'Conc.Esp':<10} {'FreqDom':<8} {'Estado':<25}")
    lines.append(f"  {'-'*5} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*25}")
    for tr in sim.time_steps:
        lines.append(
            f"  {tr.step:<5} {tr.time_value:<12.6f} {tr.fidelity:<12.6f} "
            f"{tr.shannon_entropy:<10.6f} {tr.purity:<10.6f} {tr.spectral_concentration:<10.6f} "
            f"{tr.dominant_frequency:<8} {tr.measured_state:<25}"
        )

    lines.append("")
    if sim.time_steps and sim.time_steps[0].quantum_counts:
        lines.append("  Conteos (paso 0, estado inicial):")
        total_s = sum(sim.time_steps[0].quantum_counts.values())
        for state, count in sorted(sim.time_steps[0].quantum_counts.items()):
            pct = count / total_s * 100
            if pct > 1.0:
                lines.append(f"    |{state}>: {count} ({pct:.1f}%)")

    last = sim.time_steps[-1] if sim.time_steps else None
    if last and last.quantum_counts:
        lines.append("  Conteos (paso final):")
        total_s = sum(last.quantum_counts.values())
        for state, count in sorted(last.quantum_counts.items()):
            pct = count / total_s * 100
            if pct > 1.0:
                lines.append(f"    |{state}>: {count} ({pct:.1f}%)")

    return "\n".join(lines)


def sim_to_csv(all_sims: list[SegmentSimulation]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "segmento", "n_qubits", "dim_hilbert",
        "strain_mean", "strain_std", "escala_energia",
        "paso", "tiempo",
        "fidelidad", "entropia_shannon", "pureza",
        "concentracion_espectral", "frecuencia_dominante", "estado_medido"
    ])
    for sim in all_sims:
        for tr in sim.time_steps:
            writer.writerow([
                sim.segment_index, sim.n_qubits, sim.hilbert_dim,
                f"{sim.mean_strain:.6e}", f"{sim.std_strain:.6e}",
                f"{sim.hamiltonian_energy_scale:.6e}",
                tr.step, f"{tr.time_value:.6e}",
                f"{tr.fidelity:.6f}", f"{tr.shannon_entropy:.6f}",
                f"{tr.purity:.6f}", f"{tr.spectral_concentration:.6f}",
                tr.dominant_frequency, tr.measured_state,
            ])

    writer.writerow([])
    writer.writerow(["segmento", "paso", "estado_cuantico", "conteo"])
    for sim in all_sims:
        for tr in sim.time_steps:
            if tr.quantum_counts:
                for state, count in tr.quantum_counts.items():
                    writer.writerow([sim.segment_index, tr.step, f"|{state}>", count])

    return output.getvalue()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    hdf5_path = os.path.join(base_dir, "H-H1_GWOSC_O4b_4KHZ_R1-1421381632-4096.hdf5")
    if not os.path.exists(hdf5_path):
        gwf_path = os.path.join(base_dir, "H-H1_GWOSC_O4b_4KHZ_R1-1421381632-4096.gwf")
        if os.path.exists(gwf_path):
            hdf5_path = gwf_path
        else:
            print("ERROR: No se encuentran archivos de datos GW")
            sys.exit(1)

    n_qubits = 8
    max_segments = 10
    n_time_steps = 16
    shots = 4096

    print("=" * 70)
    print("SIMULACION CUANTICA: EVOLUCION TEMPORAL DE ESTADOS GW")
    print("=" * 70)
    print(f"Config: {n_qubits} qubits, {max_segments} segmentos, {n_time_steps} pasos, {shots} shots")
    print()

    strain = load_strain_data(hdf5_path)
    segments = strain_segments(strain, n_qubits, max_segments)
    print(f"Total segmentos a simular: {len(segments)}\n")

    all_sims: list[SegmentSimulation] = []
    for i, (start_idx, end_idx, seg) in enumerate(segments):
        print(f"--- Simulacion segmento {i} ({len(seg)} muestras) ---")
        t0 = time.perf_counter()
        sim = simulate_segment(seg, i, n_time_steps=n_time_steps, shots=shots)
        elapsed = time.perf_counter() - t0
        print(f"  Tiempo de simulacion: {elapsed:.3f}s")
        print()
        print(sim_to_text(sim))
        print()
        all_sims.append(sim)

    print("=" * 70)
    print("RESUMEN GLOBAL DE LA SIMULACION")
    print("=" * 70)

    initial_s = np.mean([s.time_steps[0].shannon_entropy for s in all_sims])
    final_s = np.mean([s.time_steps[-1].shannon_entropy for s in all_sims])
    initial_p = np.mean([s.time_steps[0].purity for s in all_sims])
    final_p = np.mean([s.time_steps[-1].purity for s in all_sims])
    final_f = np.mean([s.time_steps[-1].fidelity for s in all_sims])
    print(f"Entropia media inicial: {initial_s:.6f}")
    print(f"Entropia media final:   {final_s:.6f}")
    print(f"Pureza media inicial:   {initial_p:.6f}")
    print(f"Pureza media final:     {final_p:.6f}")
    print(f"Fidelidad media final:  {final_f:.6f}")

    csv_data = sim_to_csv(all_sims)
    csv_filename = "simulacion_cuantica_resultados.csv"
    with open(csv_filename, "w") as f:
        f.write(csv_data)
    print(f"\nResultados exportados a: {csv_filename}")


if __name__ == "__main__":
    main()
