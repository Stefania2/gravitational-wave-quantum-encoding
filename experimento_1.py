from __future__ import annotations

import csv
import cmath
import io
import math
import sys
import traceback
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

HISTORY = [
    "la semilla fue sembrada",
    "la raiz busco agua",
    "el tallo encontro luz",
    "la flor se abrio",
    "el fruto guardo memoria",
    "la semilla cayo otra vez",
    "la tierra recordo el ciclo",
    "el comienzo regreso",
]

CURRENT_TIME = 11
FUTURE_JUMP = 5
REGISTER_BITS = 6
BOLTZMANN_CONSTANT = 1.380649e-23
LIGHT_SPEED = 299_792_458
GRAVITATIONAL_CONSTANT = 6.67430e-11
REDUCED_PLANCK_CONSTANT = 1.054571817e-34
PLANCK_LENGTH = 1.616255e-35
PLANCK_ENERGY = 1.9561e9

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


@dataclass(frozen=True)
class ExternalAgent:
    name: str
    entry_time: int
    mass_kg: float
    velocity_fraction_c: float
    coherence: float


@dataclass(frozen=True)
class SimulationResult:
    agent: ExternalAgent
    current_time: int
    future_jump: int
    cycle_size: int
    effective_time: int
    effective_jump: int
    cycle_period: int
    dominant_frequency: int
    event_probabilities: list[dict[str, float]]
    future_branches: list[dict[str, float]]
    qft_spectrum: list[dict[str, float]]
    information_area: float
    information_entropy: float
    probability_entropy: float
    state_distance: float
    spectral_concentration: float
    timeline: list[str]
    measured_state: str
    qiskit_used: bool
    quantum_counts: dict[str, int] | None


def cycle_index(time: int, cycle_size: int) -> int:
    return time % cycle_size


def planck_area() -> float:
    return GRAVITATIONAL_CONSTANT * REDUCED_PLANCK_CONSTANT / LIGHT_SPEED**3


def information_area_from_memory(memory_index: int) -> float:
    memory_units = memory_index + 1
    return 4 * memory_units * planck_area()


def information_entropy(area: float) -> float:
    numerator = BOLTZMANN_CONSTANT * LIGHT_SPEED**3 * area
    denominator = 4 * GRAVITATIONAL_CONSTANT * REDUCED_PLANCK_CONSTANT
    return numerator / denominator


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def event_text_signature(text: str) -> int:
    return sum(text.encode("utf-8"))


def event_weight(event: str) -> float:
    return float((event_text_signature(event) % 16) + 1)


def event_state_vector(history: list[str], agent: ExternalAgent) -> list[complex]:
    mass_scale = clamp(math.log1p(agent.mass_kg) / 4.0, 0.0, 10.0)
    velocity_scale = clamp(agent.velocity_fraction_c, 0.0, 0.999999)
    coherence = clamp(agent.coherence, 0.0, 1.0)

    amplitudes: list[complex] = []
    for index, event in enumerate(history):
        signature = event_text_signature(event)
        magnitude = 0.15 + 0.85 * (((signature >> (index % 5)) & 0x1F) + 1) / 32.0
        event_mass_factor = 1.0 + 0.18 * mass_scale * (((signature >> 2) & 0x07) + 1) / 8.0
        event_vel_factor = 1.0 + 0.24 * velocity_scale * (((signature >> 4) & 0x03) + 1) / 4.0
        entry_factor = 1.0 + 0.10 * abs(agent.entry_time) / 10.0 * (((signature >> 1) & 0x03) + 1) / 4.0
        magnitude *= event_mass_factor * event_vel_factor * entry_factor
        magnitude *= 0.5 + 0.5 * coherence
        phase = (signature % 17) / 17.0 * 2 * math.pi
        phase += agent.entry_time * 0.25
        phase += mass_scale * 0.20 * (((signature >> 3) & 0x03) + 1) / 4.0
        phase += velocity_scale * 2 * math.pi
        phase *= 0.4 + 0.6 * coherence
        amplitudes.append(cmath.rect(magnitude, phase))

    norm = math.sqrt(sum(abs(amplitude) ** 2 for amplitude in amplitudes))
    if norm == 0:
        return [complex(1 / math.sqrt(len(history)), 0) for _ in history]

    return [amplitude / norm for amplitude in amplitudes]


def event_energies(agent: ExternalAgent) -> list[float]:
    mass_scale = max(0.0, agent.mass_kg / 50.0)
    velocity_scale = clamp(agent.velocity_fraction_c, 0.0, 0.999999)
    energies: list[float] = []
    for event in HISTORY:
        signature = event_text_signature(event)
        local = ((signature % 64) + 20) * (1.0 + 0.3 * mass_scale)
        energies.append(local * (1.0 + 0.8 * velocity_scale))
    return energies


def phase_evolution(psi: list[complex], energies: list[float], agent: ExternalAgent) -> list[complex]:
    phase_scale = 0.2 + 0.8 * clamp(agent.coherence, 0.0, 1.0)
    dt = 1.0 + agent.entry_time * 0.2
    return [
        amplitude * cmath.exp(-1j * energy * phase_scale * dt * 0.02)
        for amplitude, energy in zip(psi, energies)
    ]


def qft_transform(psi: list[complex]) -> list[complex]:
    n = len(psi)
    factor = 1 / math.sqrt(n)
    return [
        factor * sum(psi[j] * cmath.exp(2j * math.pi * j * k / n) for j in range(n))
        for k in range(n)
    ]


def detect_cycle_period(spectrum: list[complex]) -> tuple[int, int]:
    n = len(spectrum)
    magnitudes = [abs(amplitude) for amplitude in spectrum]
    if not any(magnitudes):
        return n, 0
    dominant = max(range(n), key=lambda i: magnitudes[i])
    period = n // (dominant or 1)
    return period, dominant


def probabilities_from_state(psi: list[complex]) -> list[float]:
    raw = [abs(amplitude) ** 2 for amplitude in psi]
    total = sum(raw)
    if total == 0:
        return [1.0 / len(raw) for _ in raw]
    return [value / total for value in raw]


def shannon_entropy(probabilities: list[float]) -> float:
    return -sum(prob * math.log2(prob) for prob in probabilities if prob > 0)


def state_distance(before: list[complex], after: list[complex]) -> float:
    return math.sqrt(sum(abs(after_amp - before_amp) ** 2 for before_amp, after_amp in zip(before, after)))


def spectral_concentration(spectrum: list[complex]) -> float:
    magnitudes = [abs(amplitude) ** 2 for amplitude in spectrum]
    total = sum(magnitudes)
    if total == 0:
        return 0.0
    return max(magnitudes) / total


def top_future_branches(probabilities: list[float], history: list[str], count: int = 3) -> list[dict[str, float]]:
    indexed = sorted(
        enumerate(probabilities),
        key=lambda item: item[1],
        reverse=True,
    )[:count]
    return [
        {"event": history[index], "probability": probability}
        for index, probability in indexed
    ]


def qft_spectrum_summary(spectrum: list[complex]) -> list[dict[str, float]]:
    return [
        {"frequency": index, "magnitude": abs(amplitude)}
        for index, amplitude in enumerate(spectrum)
    ]


def lorentz_factor(beta: float) -> float:
    beta = clamp(abs(beta), 0.0, 0.999999)
    return 1 / math.sqrt(1 - beta**2)


def kinetic_energy(agent: ExternalAgent) -> float:
    gamma = lorentz_factor(agent.velocity_fraction_c)
    return (gamma - 1) * agent.mass_kg * LIGHT_SPEED**2


def gravitational_radius(agent: ExternalAgent) -> float:
    return 2 * GRAVITATIONAL_CONSTANT * agent.mass_kg / LIGHT_SPEED**2


def perturbation_angles(agent: ExternalAgent, bit_count: int) -> list[float]:
    energy_ratio = max(0.0, kinetic_energy(agent)) / PLANCK_ENERGY
    gravity_ratio = max(0.0, gravitational_radius(agent)) / PLANCK_LENGTH
    coherence = clamp(agent.coherence, 0.0, 1.0)
    entry_phase = (abs(agent.entry_time) + 1) / 10
    energy_phase = math.log1p(energy_ratio)
    gravity_phase = math.log1p(gravity_ratio)
    decoherence_phase = (1 - coherence) * math.pi

    angles: list[float] = []
    for index in range(bit_count):
        phase = (
            energy_phase / (index + 1)
            + gravity_phase * (index + 1) / bit_count
            + entry_phase
            + decoherence_phase
        )
        angles.append((phase % (2 * math.pi)) * coherence)
    return angles


def build_cycle_operator(agent: ExternalAgent, history: list[str]) -> tuple[list[complex], list[complex], list[float]]:
    psi0 = event_state_vector(history, agent)
    energies = event_energies(agent)
    evolved = phase_evolution(psi0, energies, agent)
    return psi0, evolved, energies


def run_quantum_circuit() -> dict[str, int]:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    print("--- CIRCUITO CUÁNTICO CREADO ---")
    print(qc)

    aer_backend = AerSimulator()
    pm_aer = generate_preset_pass_manager(optimization_level=1, backend=aer_backend)
    isa_circuit_aer = pm_aer.run(qc)
    sampler_aer = SamplerV2(mode=aer_backend)
    job_aer = sampler_aer.run([isa_circuit_aer])
    result_aer = job_aer.result()
    counts = result_aer[0].data.meas.get_counts()

    print("Resultados del Simulador Local:", counts)
    return counts


def decode_logical_state(value: int) -> str:
    return TEMPORAL_CODES.get(value % 16, "estado desconocido")


def run_simulation(agent: ExternalAgent, current_time: int, future_jump: int, quantum_counts: dict[str, int] | None = None) -> SimulationResult:
    cycle_size = len(HISTORY)
    effective_time = current_time + agent.entry_time
    effective_jump = future_jump + int(agent.coherence * 10)
    initial_state, evolved_state, energies = build_cycle_operator(agent, HISTORY)
    spectrum = qft_transform(evolved_state)
    cycle_period, dominant_frequency = detect_cycle_period(spectrum)
    future_branches = simulate_future_branches(evolved_state, HISTORY)
    probabilities = probabilities_from_state(evolved_state)
    event_probabilities = [
        {"event": event, "probability": prob}
        for event, prob in zip(HISTORY, probabilities)
    ]
    area = information_area_from_memory(cycle_period)
    return SimulationResult(
        agent=agent,
        current_time=current_time,
        future_jump=future_jump,
        cycle_size=cycle_size,
        effective_time=effective_time,
        effective_jump=effective_jump,
        cycle_period=cycle_period,
        dominant_frequency=dominant_frequency,
        event_probabilities=event_probabilities,
        future_branches=future_branches,
        qft_spectrum=qft_spectrum_summary(spectrum),
        information_area=area,
        information_entropy=information_entropy(area),
        probability_entropy=shannon_entropy(probabilities),
        state_distance=state_distance(initial_state, evolved_state),
        spectral_concentration=spectral_concentration(spectrum),
        timeline=HISTORY,
        measured_state=decode_logical_state(dominant_frequency),
        qiskit_used=True,
        quantum_counts=quantum_counts,
    )


def simulate_future_branches(psi: list[complex], history: list[str]) -> list[dict[str, float]]:
    probabilities = probabilities_from_state(psi)
    return top_future_branches(probabilities, history)


def simulation_to_text(result: SimulationResult) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("RESULTADOS DE LA SIMULACION CUANTICA")
    lines.append("=" * 60)
    lines.append(f"Agente: {result.agent.name}")
    lines.append(f"Masa: {result.agent.mass_kg} kg")
    lines.append(f"Fraccion de velocidad c: {result.agent.velocity_fraction_c}")
    lines.append(f"Coherencia: {result.agent.coherence}")
    lines.append(f"Tiempo de entrada: {result.agent.entry_time}")
    lines.append("")
    lines.append(f"Tiempo base: {result.current_time}")
    lines.append(f"Salto base: {result.future_jump}")
    lines.append(f"Indice logico efectivo: {result.effective_time}")
    lines.append(f"Desplazamiento logico efectivo: {result.effective_jump}")
    lines.append(f"Periodo detectado (QFT): {result.cycle_period}")
    lines.append(f"Frecuencia dominante: {result.dominant_frequency}")
    lines.append(f"Interpretacion logica: {result.measured_state}")
    lines.append("")

    if result.quantum_counts:
        lines.append("Resultados del circuito cuantico (Bell state):")
        total = sum(result.quantum_counts.values())
        for state, count in sorted(result.quantum_counts.items()):
            pct = count / total * 100
            lines.append(f"  |{state}>: {count} ({pct:.1f}%)")
        lines.append("")

    lines.append("Espectro QFT:")
    lines.append(f"  {'Freq':<6} {'Magnitud':<12}")
    for item in result.qft_spectrum:
        lines.append(f"  {item['frequency']:<6} {item['magnitude']:<12.6f}")
    lines.append("")

    lines.append("Ramas logicas dominantes:")
    lines.append(f"  {'Evento':<35} {'Probabilidad':<12}")
    for branch in result.future_branches:
        lines.append(f"  {branch['event']:<35} {branch['probability']:<12.6f}")
    lines.append("")

    lines.append(f"Area analogica de informacion: {result.information_area:.6e}")
    lines.append(f"Entropia analogica de informacion: {result.information_entropy:.6e}")
    lines.append(f"Entropia de probabilidad: {result.probability_entropy:.6f}")
    lines.append(f"Distancia entre estados: {result.state_distance:.6f}")
    lines.append(f"Concentracion espectral: {result.spectral_concentration:.6f}")
    lines.append(f"Qiskit usado: {'si' if result.qiskit_used else 'no'}")
    lines.append("=" * 60)

    return "\n".join(lines)


def simulation_to_csv(result: SimulationResult) -> str:
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Campo", "Valor"])
    writer.writerow(["Nombre agente", result.agent.name])
    writer.writerow(["Masa (kg)", result.agent.mass_kg])
    writer.writerow(["Fraccion velocidad c", result.agent.velocity_fraction_c])
    writer.writerow(["Coherencia", result.agent.coherence])
    writer.writerow(["Tiempo base", result.current_time])
    writer.writerow(["Salto base", result.future_jump])
    writer.writerow(["Indice logico efectivo", result.effective_time])
    writer.writerow(["Desplazamiento logico efectivo", result.effective_jump])
    writer.writerow(["Periodo detectado (QFT)", result.cycle_period])
    writer.writerow(["Frecuencia dominante", result.dominant_frequency])
    writer.writerow(["Interpretacion logica", result.measured_state])
    writer.writerow(["Area analogica de informacion", f"{result.information_area:.6e}"])
    writer.writerow(["Entropia analogica de informacion", f"{result.information_entropy:.6e}"])
    writer.writerow(["Entropia de probabilidad", f"{result.probability_entropy:.6f}"])
    writer.writerow(["Distancia entre estados", f"{result.state_distance:.6f}"])
    writer.writerow(["Concentracion espectral", f"{result.spectral_concentration:.6f}"])
    writer.writerow(["Qiskit usado", "si" if result.qiskit_used else "no"])

    if result.quantum_counts:
        writer.writerow([])
        writer.writerow(["Estado cuantico", "Conteo", "Probabilidad"])
        total = sum(result.quantum_counts.values())
        for state, count in sorted(result.quantum_counts.items()):
            writer.writerow([f"|{state}>", count, f"{count/total:.4f}"])

    writer.writerow([])
    writer.writerow(["Evento", "Probabilidad"])
    for item in result.event_probabilities:
        writer.writerow([item["event"], f"{item['probability']:.6f}"])

    writer.writerow([])
    writer.writerow(["Rama logica dominante", "Probabilidad"])
    for branch in result.future_branches:
        writer.writerow([branch["event"], f"{branch['probability']:.6f}"])

    return output.getvalue()


if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENTO CUANTICO: CIRCUITO BELL + SIMULACION CICLICA")
    print("=" * 60)

    counts = run_quantum_circuit()

    agent = ExternalAgent(
        name="observador",
        entry_time=0,
        mass_kg=70.0,
        velocity_fraction_c=0.01,
        coherence=0.75,
    )

    result = run_simulation(agent, CURRENT_TIME, FUTURE_JUMP, quantum_counts=counts)

    print("\n" + simulation_to_text(result))

    csv_data = simulation_to_csv(result)
    filename = "resultado_simulacion.csv"
    with open(filename, "w") as f:
        f.write(csv_data)
    print(f"\nResultados exportados a: {filename}")

    print("\n¿Enviar tambien a IBM Quantum real? (s/n): ", end="")
    try:
        respuesta = input().strip().lower()
        if respuesta == "s":
            print("\nConectando con IBM Quantum...")
            service = QiskitRuntimeService()
            real_backend = service.least_busy(operational=True, simulator=False)
            print(f"Enviando trabajo a: {real_backend.name}")

            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure_all()

            pm_real = generate_preset_pass_manager(optimization_level=1, backend=real_backend)
            isa_circuit_real = pm_real.run(qc)
            sampler_ibm = SamplerV2(mode=real_backend)
            job_ibm = sampler_ibm.run([isa_circuit_real])
            print(f"ID del trabajo: {job_ibm.job_id()}")
            print("Trabajo enviado a la nube de IBM.")
    except (EOFError, KeyboardInterrupt):
        print()
