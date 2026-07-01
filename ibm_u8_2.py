from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator

import matplotlib.pyplot as plt

service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    instance='crn:v1:bluemix:public:quantum-computing:us-east:a/7c2b73157c32465594552e57ae8036e5:bb853402-dadc-4f98-93a6-19d4802ec4dc::'
)

job = service.job('d8u2bpcbp3hs7385b5i0')

result = job.result()

counts = result[0].data.meas.get_counts()

print("\nResultados:")
print(counts)

print(job.backend()) #lenguaje python
print(job.backend()) #Estado del trabajo:
print(result[0].metadata) #print(result[0].metadata)

counts = result[0].data.meas.get_counts()
print(counts) 



counts = {
    '00': 2032,
    '11': 1898,
    '10': 96,
    '01': 70
}

fig = plot_histogram(counts)

plt.title(
    "Resultados experimentales en IBM Fez\nEstado Bell generado en hardware cuántico"
)

plt.ylabel("Número de mediciones")
plt.xlabel("Estado computacional")

plt.tight_layout()

plt.savefig(
    "bell_state_ibm_fez.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#Probabilidades de los estados computacionales

shots = sum(counts.values())

probabilidades = {
    estado: valor/shots
    for estado, valor in counts.items()
}

print(probabilidades)

plot_histogram(
    probabilidades,
    title="Distribución de Probabilidades"
)
plt.show()


# ==========================================
# DATOS GW250120_042414
# ==========================================

data = {
    "strain_mean":[
        -3.495589e-18,-3.635966e-18,-3.754076e-18,-3.849522e-18,
        -3.921628e-18,-3.969897e-18,-3.994416e-18,-3.995013e-18,
        -3.971378e-18,-3.924374e-18
    ],
    "strain_std":[
        4.150357e-20,3.584759e-20,2.997259e-20,2.389001e-20,
        1.767741e-20,1.140305e-20,5.267535e-21,4.304334e-21,
        1.051384e-20,1.763883e-20
    ],
    "entropia":[
        3.638106,3.724959,3.718721,3.810353,3.786191,
        3.685730,3.278494,4.354649,3.035475,3.770848
    ],
    "pureza":[
        0.147321,0.134473,0.123224,0.101903,0.095188,
        0.107479,0.139447,0.073194,0.194901,0.115334
    ],
    "conc_espectral":[
        0.092314,0.094741,0.089806,0.088080,0.082020,
        0.072493,0.062695,0.119821,0.060381,0.088175
    ]
}

df = pd.DataFrame(data)

# ==========================================
# SEGMENTO A ANALIZAR
# ==========================================

segmento = 0

# ==========================================
# NORMALIZACIÓN A [0, pi]
# ==========================================

def norm_pi(x, xmin, xmax):
    return np.pi * (x - xmin) / (xmax - xmin)

features = [
    "strain_mean",
    "strain_std",
    "entropia",
    "pureza",
    "conc_espectral"
]

angulos = []

for col in features:

    theta = norm_pi(
        df.loc[segmento, col],
        df[col].min(),
        df[col].max()
    )

    angulos.append(theta)

print("\nÁngulos obtenidos:")
for i,a in enumerate(angulos):
    print(f"q{i}: {a:.4f}")

# ==========================================
# CIRCUITO CUÁNTICO
# ==========================================

nq = len(angulos)

qc = QuantumCircuit(nq, nq)

# Angle Encoding

for i,theta in enumerate(angulos):
    qc.ry(theta, i)

# Entrelazamiento

for i in range(nq-1):
    qc.cx(i, i+1)

qc.barrier()

qc.measure_all()

# ==========================================
# DIBUJO
# ==========================================

qc.draw("mpl")
plt.show()

# ==========================================
# SIMULACIÓN
# ==========================================

backend = AerSimulator()

job = backend.run(qc, shots=4096)

result = job.result()

counts = result.get_counts()

print("\nResultados:")
print(counts)

plot_histogram(counts)
plt.show()