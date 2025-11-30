#!/usr/bin/env python
"""
Demo 03 - QAOA MaxCut (Cirq)

This demo:
- Builds a 5-node graph
- Constructs a QAOA circuit for MaxCut
- Optimizes the cost function
- Saves JSON results and a cost convergence plot
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

import cirq


def build_graph():
    """Simple 5-node ring graph."""
    nodes = list(range(5))
    edges = [(i, (i + 1) % 5) for i in nodes]
    return nodes, edges


def maxcut_cost(bitstring, edges):
    """Compute MaxCut cost for a given bitstring."""
    cost = 0
    for u, v in edges:
        if bitstring[u] != bitstring[v]:
            cost += 1
    return cost


def qaoa_circuit(gamma, beta, qubits, edges):
    """Build a QAOA layer for MaxCut in modern Cirq syntax."""
    circuit = cirq.Circuit()

    # Initial state: Hadamard on all qubits
    circuit.append(cirq.H.on_each(*qubits))

    # Cost Hamiltonian: ZZ interactions
    for u, v in edges:
        circuit.append(cirq.ZZ(qubits[u], qubits[v]) ** gamma)

    # Mixer Hamiltonian: RX rotations
    for q in qubits:
        circuit.append(cirq.rx(2 * beta)(q))

    # Measurement key "m"
    circuit.append(cirq.measure(*qubits, key="m"))

    return circuit


def expectation(simulator, gamma, beta, qubits, edges, shots=2000):
    """Estimate expected MaxCut value."""
    circuit = qaoa_circuit(gamma, beta, qubits, edges)
    result = simulator.run(circuit, repetitions=shots)

    bitstrings = result.measurements["m"]

    total_cost = 0
    for bits in bitstrings:
        bits = bits.tolist()
        total_cost += maxcut_cost(bits, edges)

    return total_cost / shots


def run_qaoa(output_dir, steps=25, stepsize=0.2):
    """Gradient-ascent QAOA optimization loop."""
    os.makedirs(output_dir, exist_ok=True)

    nodes, edges = build_graph()
    qubits = cirq.LineQubit.range(len(nodes))
    simulator = cirq.Simulator()

    # Initial parameters
    gamma = 0.5
    beta = 0.5

    history = []

    for step in range(steps):
        # Compute baseline cost
        base_cost = expectation(simulator, gamma, beta, qubits, edges)

        # Finite-difference gradients
        dg = (expectation(simulator, gamma + 1e-2, beta, qubits, edges) - base_cost) / 1e-2
        db = (expectation(simulator, gamma, beta + 1e-2, qubits, edges) - base_cost) / 1e-2

        # Gradient ascent (maximize cost)
        gamma += stepsize * dg
        beta += stepsize * db

        history.append({"step": step, "cost": base_cost})

        print(f"[step {step}] cost={base_cost:.4f}, gamma={gamma:.4f}, beta={beta:.4f}")

    # Save JSON results
    json_path = os.path.join(output_dir, "results_demo03_qaoa.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)

    # Plot convergence
    plt.figure(figsize=(8, 5))
    plt.plot(
        [h["step"] for h in history],
        [h["cost"] for h in history],
        marker="o"
    )
    plt.xlabel("Iteration")
    plt.ylabel("Expected Cut Value")
    plt.title("Demo 03 - QAOA MaxCut Convergence (Cirq)")
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "cost_convergence_demo03.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved JSON to {json_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results/demo03")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--stepsize", type=float, default=0.2)
    args = parser.parse_args()

    run_qaoa(args.output_dir, args.steps, args.stepsize)

