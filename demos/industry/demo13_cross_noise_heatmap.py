#!/usr/bin/env python
"""
Demo 13 — Cross-Noise Robustness Heatmap (Qiskit, Cirq, PennyLane)

This demo:
- Builds a simple 2-qubit parity circuit.
- Evaluates expectation <ZZ> across multiple noise levels.
- Compares Qiskit Aer, Cirq, and PennyLane.
- Produces:
    1. noise_matrix_demo13.json
    2. noise_heatmap_demo13.png
    3. noise_curves_demo13.png
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# ============================================================
# Reference rotation angle
# ============================================================

def theta():
    return pi/4


# ============================================================
# QISKIT IMPLEMENTATION
# ============================================================

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

def build_qiskit_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0,1)
    qc.ry(theta(), 0)
    qc.cx(0,1)
    qc.measure_all()
    return qc

def parity_from_counts(counts):
    shots=sum(counts.values())
    exp=0
    for bits,c in counts.items():
        parity=bits.count("1")%2
        sign = 1 if parity==0 else -1
        exp+=sign*c/shots
    return exp

def qiskit_expectation(noise_level):
    qc=build_qiskit_circuit()

    if noise_level == 0:
        sim = AerSimulator()
    else:
        nm=NoiseModel()
        nm.add_all_qubit_quantum_error(
            depolarizing_error(noise_level,1),
            ["h","ry"]
        )
        nm.add_all_qubit_quantum_error(
            depolarizing_error(noise_level,2),
            ["cx"]
        )
        sim = AerSimulator(noise_model=nm)

    result=sim.run(qc,shots=3000).result()
    counts=result.get_counts()
    return parity_from_counts(counts)


# ============================================================
# CIRQ IMPLEMENTATION
# ============================================================

import cirq

def build_cirq_circuit():
    q0,q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0,q1),
        cirq.ry(theta())(q0),
        cirq.CNOT(q0,q1),
    )
    return circuit,(q0,q1)

def cirq_expectation(noise_level):
    circuit,(q0,q1)=build_cirq_circuit()

    if noise_level == 0:
        sim = cirq.Simulator()
        result = sim.simulate(circuit)
        final = result.final_state_vector

        Z = np.array([[1,0],[0,-1]])
        ZZ = np.kron(Z,Z)
        return float(np.conj(final) @ (ZZ @ final))

    # Apply depolarizing noise
    noisy = circuit.with_noise(cirq.depolarize(noise_level))

    # 🔧 IMPORTANT: add explicit measurement gate
    noisy += cirq.Circuit(cirq.measure(q0, q1, key="m"))

    sim = cirq.DensityMatrixSimulator()
    result = sim.run(noisy, repetitions=3000)

    bits = result.measurements["m"]
    exp=0
    for b in bits:
        parity = int(b[0]) ^ int(b[1])
        sign = 1 if parity==0 else -1
        exp += sign/len(bits)
    return exp


# ============================================================
# PENNYLANE IMPLEMENTATION
# ============================================================

import pennylane as qml

def pennylane_expectation(noise_level):
    if noise_level == 0:
        dev = qml.device("default.qubit", wires=2)
    else:
        dev = qml.device("default.mixed", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.H(0)
        qml.CNOT([0,1])
        qml.RY(theta(), wires=0)
        qml.CNOT([0,1])
        if noise_level > 0:
            qml.DepolarizingChannel(noise_level, wires=0)
            qml.DepolarizingChannel(noise_level, wires=1)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    return float(circuit())


# ============================================================
# MAIN DEMO
# ============================================================

def run_demo(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    noise_levels=[0,0.02,0.05,0.10,0.20]

    matrix={
        "qiskit":[],
        "cirq":[],
        "pennylane":[]
    }

    print("\n=== Running Cross-Noise Robustness Benchmark (Demo 13) ===\n")

    for nl in noise_levels:
        print(f"Noise level: {nl}")

        q_val = qiskit_expectation(nl)
        c_val = cirq_expectation(nl)
        p_val = pennylane_expectation(nl)

        matrix["qiskit"].append(q_val)
        matrix["cirq"].append(c_val)
        matrix["pennylane"].append(p_val)

        print(f"  Qiskit:     {q_val:.4f}")
        print(f"  Cirq:       {c_val:.4f}")
        print(f"  PennyLane:  {p_val:.4f}\n")

    # Save JSON
    json_path=os.path.join(output_dir,"noise_matrix_demo13.json")
    with open(json_path,"w") as f:
        json.dump({"noise_levels":noise_levels,"matrix":matrix},f,indent=2)

    # Heatmap
    data=np.array([matrix["qiskit"],matrix["cirq"],matrix["pennylane"]])

    plt.figure(figsize=(8,4))
    plt.imshow(data,cmap="viridis",aspect="auto")
    plt.colorbar(label="Expectation <ZZ>")
    plt.yticks([0,1,2],["Qiskit","Cirq","PennyLane"])
    plt.xticks(range(len(noise_levels)),noise_levels)
    plt.title("Demo 13 — Cross-Noise Robustness Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"noise_heatmap_demo13.png"))
    plt.close()

    # Noise Drift Curves
    plt.figure(figsize=(8,4))
    for fw in ["qiskit","cirq","pennylane"]:
        plt.plot(noise_levels,matrix[fw],marker="o",label=fw)
    plt.title("Demo 13 — Noise Drift Curves")
    plt.xlabel("Noise Level")
    plt.ylabel("Expectation <ZZ>")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"noise_curves_demo13.png"))
    plt.close()

    print("\n===== DEMO 13 SUMMARY =====")
    print(matrix)
    print(f"\nSaved JSON: {json_path}")
    print("Saved heatmap + curves in:", output_dir)


# CLI
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output_dir",default="results/demo13")
    args=parser.parse_args()
    run_demo(args.output_dir)
