#!/usr/bin/env python
"""
Demo 10 — Energy Grid Optimization using QAOA (Cirq)

This industry demo:
- Creates a small 4-node microgrid with weighted transmission lines.
- Formulates the load-balancing / cut-minimization as a MaxCut problem.
- Uses QAOA to approximate the optimal grid partition.
- Compares QAOA to classical brute force.
- Saves results in JSON + PNG.

Outputs:
- results/demo10/results_demo10_energy.json
- results/demo10/qaoa_energy_plot.png
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

import cirq


# ============================================================
# Microgrid Definition
# ============================================================

def build_microgrid():
    """
    Returns:
    nodes: 4 microgrid nodes
    edges: weighted connections (as (u, v, weight))
    """
    nodes = list(range(4))

    # Example microgrid transmission costs
    # Lower cost = desirable, higher cost = strain or loss
    edges = [
        (0, 1, 1.0),   # generator-to-load
        (1, 2, 2.0),   # load-to-load
        (2, 3, 1.5),   # load-to-generator
        (0, 3, 2.5),   # generator-to-generator loop
    ]
    return nodes, edges


# ============================================================
# MaxCut Cost Function (classical)
# ============================================================

def classical_cost(bitstring, edges):
    cost = 0
    for (u, v, w) in edges:
        if bitstring[u] != bitstring[v]:
            cost += w
    return cost


def brute_force_solution(nodes, edges):
    n = len(nodes)
    best_cost = -np.inf
    best_bits = None

    for b in range(2 ** n):
        bits = [(b >> i) & 1 for i in range(n)]
        c = classical_cost(bits, edges)
        if c > best_cost:
            best_cost = c
            best_bits = bits
    return best_bits, best_cost


# ============================================================
# QAOA Circuit
# ============================================================

def qaoa_layer(gamma, beta, qubits, edges):
    circuit = cirq.Circuit()

    # Cost Hamiltonian (ZZ terms)
    for (u, v, w) in edges:
        circuit.append(cirq.ZZ(qubits[u], qubits[v]) ** (gamma * w))

    # Mixer Hamiltonian (RX)
    for q in qubits:
        circuit.append(cirq.rx(2 * beta)(q))

    return circuit


def build_qaoa_circuit(gammas, betas, qubits, edges):
    """Full QAOA circuit."""
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(qubits))  # Uniform superposition

    for gamma, beta in zip(gammas, betas):
        circuit += qaoa_layer(gamma, beta, qubits, edges)

    circuit.append(cirq.measure(*qubits, key="m"))
    return circuit


# ============================================================
# Expectation (Cost Estimation)
# ============================================================

def expected_cost(simulator, circuit, edges, shots=3000):
    result = simulator.run(circuit, repetitions=shots)
    counts = result.measurements["m"]

    total_cost = 0
    for bits in counts:
        bits = bits.tolist()
        total_cost += classical_cost(bits, edges)
    return total_cost / shots


# ============================================================
# Simple Gradient-Free Optimization
# ============================================================

def optimize_qaoa(p, qubits, edges, steps=20, lr=0.2):
    """
    p: QAOA depth
    Returns optimized (gammas, betas)
    """

    gammas = np.random.uniform(0, 2*np.pi, p)
    betas = np.random.uniform(0, 2*np.pi, p)

    sim = cirq.Simulator()

    for step in range(steps):
        candidate_g = gammas + lr * np.random.randn(p)
        candidate_b = betas + lr * np.random.randn(p)

        circuit_candidate = build_qaoa_circuit(candidate_g, candidate_b, qubits, edges)
        circuit_current   = build_qaoa_circuit(gammas,       betas,       qubits, edges)

        c_new  = expected_cost(sim, circuit_candidate, edges)
        c_old  = expected_cost(sim, circuit_current,   edges)

        if c_new > c_old:
            gammas, betas = candidate_g, candidate_b

        if step % 5 == 0:
            print(f"[QAOA Step {step}] cost={c_old:.3f}")

    return gammas, betas


# ============================================================
# Demo Runner
# ============================================================

def run_demo(output_dir, p=1):
    os.makedirs(output_dir, exist_ok=True)

    nodes, edges = build_microgrid()
    qubits = cirq.LineQubit.range(len(nodes))

    # Classical brute force (for comparison)
    best_bits, best_cost = brute_force_solution(nodes, edges)
    print("Brute-force best:", best_bits, "cost:", best_cost)

    # QAOA parameter optimization
    gammas, betas = optimize_qaoa(p, qubits, edges, steps=20, lr=0.3)

    # Build final circuit & compute final QAOA cost
    final_circuit = build_qaoa_circuit(gammas, betas, qubits, edges)
    sim = cirq.Simulator()
    qaoa_cost = expected_cost(sim, final_circuit, edges)

    results = {
        "classical_best_bits": best_bits,
        "classical_best_cost": float(best_cost),
        "qaoa_cost": float(qaoa_cost),
        "qaoa_gammas": gammas.tolist(),
        "qaoa_betas": betas.tolist(),
    }

    # Save JSON
    json_path = os.path.join(output_dir, "results_demo10_energy.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot comparison
    labels = ["Classical Optimum", "QAOA Estimate"]
    costs = [best_cost, qaoa_cost]

    plt.figure(figsize=(6,4))
    plt.bar(labels, costs, color=["green", "orange"])
    plt.title("Demo 10 — Energy Grid Optimization: Classical vs QAOA")
    plt.ylabel("Expected Cost")
    plt.tight_layout()

    png_path = os.path.join(output_dir, "qaoa_energy_plot.png")
    plt.savefig(png_path)
    plt.close()

    print("\n===== DEMO 10 SUMMARY =====")
    print(results)
    print(f"\nSaved JSON to {json_path}")
    print(f"Saved plot to {png_path}")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/demo10")
    parser.add_argument("--p", type=int, default=1)
    args = parser.parse_args()
    run_demo(args.output_dir, p=args.p)

