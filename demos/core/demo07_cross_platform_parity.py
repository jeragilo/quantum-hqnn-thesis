#!/usr/bin/env python
"""
Demo 07 — Cross-Platform Parity Consistency Test

This demo:
- Builds the exact same 2-qubit state in Qiskit, Cirq, and PennyLane
- Computes expectation values of:
      ⟨Z0⟩, ⟨Z1⟩, ⟨Z0 Z1⟩,  ⟨X0 X1⟩
- Uses *statevector* simulation only
- Compares framework numerical drift

Outputs:
- results_demo07.json
- parity_demo07.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# ------------------------------------------------------------
# Qiskit (statevector)
# ------------------------------------------------------------
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli


def qiskit_expectations(theta):
    qc = QuantumCircuit(2)
    qc.h(1)
    qc.ry(theta, 0)
    qc.cx(0,1)

    state = Statevector.from_instruction(qc)

    exp_z0 = state.expectation_value(Pauli("ZI"))
    exp_z1 = state.expectation_value(Pauli("IZ"))
    exp_zz = state.expectation_value(Pauli("ZZ"))
    exp_xx = state.expectation_value(Pauli("XX"))

    return float(exp_z0), float(exp_z1), float(exp_zz), float(exp_xx)


# ------------------------------------------------------------
# Cirq
# ------------------------------------------------------------
import cirq

def cirq_expectations(theta):
    q0, q1 = cirq.LineQubit.range(2)

    circuit = cirq.Circuit(
        cirq.H(q1),
        cirq.ry(theta)(q0),
        cirq.CNOT(q0, q1)
    )

    sim = cirq.Simulator()
    result = sim.simulate(circuit)
    state = result.final_state_vector

    # Compute expectation values manually
    Z = np.array([[1,0],[0,-1]])
    X = np.array([[0,1],[1,0]])

    # Tensor products
    Z0 = np.kron(Z, np.eye(2))
    Z1 = np.kron(np.eye(2), Z)
    ZZ = np.kron(Z, Z)
    XX = np.kron(X, X)

    exp_z0 = float(np.conj(state) @ (Z0 @ state))
    exp_z1 = float(np.conj(state) @ (Z1 @ state))
    exp_zz = float(np.conj(state) @ (ZZ @ state))
    exp_xx = float(np.conj(state) @ (XX @ state))

    return exp_z0, exp_z1, exp_zz, exp_xx


# ------------------------------------------------------------
# PennyLane
# ------------------------------------------------------------
import pennylane as qml

def pl_expectations(theta):
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.H(1)
        qml.RY(theta, wires=0)
        qml.CNOT(wires=[0,1])
        return (
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(1)),
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
            qml.expval(qml.PauliX(0) @ qml.PauliX(1))
        )

    return tuple([float(v) for v in circuit()])


# ------------------------------------------------------------
# Demo Execution
# ------------------------------------------------------------

def run_demo(output_dir="results/demo07", theta=pi/4):
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Demo 07 — Cross-Platform Parity Consistency ===\n")

    q_z0, q_z1, q_zz, q_xx = qiskit_expectations(theta)
    c_z0, c_z1, c_zz, c_xx = cirq_expectations(theta)
    p_z0, p_z1, p_zz, p_xx = pl_expectations(theta)

    results = {
        "qiskit":  {"Z0": q_z0, "Z1": q_z1, "ZZ": q_zz, "XX": q_xx},
        "cirq":    {"Z0": c_z0, "Z1": c_z1, "ZZ": c_zz, "XX": c_xx},
        "pennylane": {"Z0": p_z0, "Z1": p_z1, "ZZ": p_zz, "XX": p_xx},
    }

    print(results)

    # Save JSON
    json_path = os.path.join(output_dir, "results_demo07.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    labels = ["Z0", "Z1", "ZZ", "XX"]
    frameworks = ["qiskit", "cirq", "pennylane"]
    colors = ["red", "blue", "green"]

    plt.figure(figsize=(10,5))
    for i, fw in enumerate(frameworks):
        vals = [results[fw][k] for k in labels]
        plt.plot(labels, vals, marker="o", color=colors[i], label=fw)

    plt.title("Demo 07 — Cross-Platform Expectation Value Consistency")
    plt.ylabel("Expectation Value")
    plt.ylim([-1.1, 1.1])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "parity_demo07.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"\nSaved JSON to {json_path}")
    print(f"Saved plot to {plot_path}")


# CLI
if __name__ == "__main__":
    run_demo()
