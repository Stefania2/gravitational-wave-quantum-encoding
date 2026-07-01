from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.visualization import (
    plot_histogram,
    plot_bloch_multivector,
    plot_state_city
)
from numpy import pi
import matplotlib.pyplot as plt

# ==================================================
# CIRCUITO (SOLO 2 QUBITS UTILIZADOS)
# ==================================================

qreg_q = QuantumRegister(2, 'q')
creg_meas = ClassicalRegister(2, 'meas')

circuit = QuantumCircuit(qreg_q, creg_meas)

circuit.rz(pi/2, qreg_q[0])
circuit.sx(qreg_q[0])
circuit.rz(pi/2, qreg_q[0])

circuit.rz(pi/2, qreg_q[1])
circuit.sx(qreg_q[1])
circuit.rz(pi/2, qreg_q[1])

circuit.cz(qreg_q[0], qreg_q[1])

circuit.rz(pi/2, qreg_q[1])
circuit.sx(qreg_q[1])
circuit.rz(pi/2, qreg_q[1])

circuit.barrier()

# ==================================================
# ESTADO ANTES DE MEDIR
# ==================================================

state = Statevector.from_instruction(circuit)

print("\n=== VECTOR DE ESTADO ===\n")
print(state)

print("\n=== PROBABILIDADES TEÓRICAS ===\n")
for estado, prob in state.probabilities_dict().items():
    print(f"{estado}: {prob:.6f}")

# ==================================================
# DIBUJAR CIRCUITO
# ==================================================

fig = circuit.draw(output='mpl')
plt.show()

# ==================================================
# ESFERAS DE BLOCH
# ==================================================

plot_bloch_multivector(state)
plt.show()

# ==================================================
# MATRIZ DE DENSIDAD
# ==================================================

rho = DensityMatrix(state)

print("\n=== MATRIZ DE DENSIDAD ===\n")
print(rho.data)

plot_state_city(rho)
plt.show()

# ==================================================
# AÑADIR MEDICIONES
# ==================================================

circuit.measure(qreg_q[0], creg_meas[0])
circuit.measure(qreg_q[1], creg_meas[1])

# ==================================================
# SIMULACIÓN
# ==================================================

simulator = AerSimulator()

compiled = transpile(circuit, simulator)

result = simulator.run(
    compiled,
    shots=10000
).result()

counts = result.get_counts()

print("\n=== RESULTADOS DE MEDICIÓN ===\n")
print(counts)

# ==================================================
# HISTOGRAMA
# ==================================================

plot_histogram(counts)
plt.show()

# ==================================================
# RESUMEN
# ==================================================

print("\n=== RESUMEN ===")
print("Qubits:", circuit.num_qubits)
print("Clásicos:", circuit.num_clbits)
print("Shots:", 10000)