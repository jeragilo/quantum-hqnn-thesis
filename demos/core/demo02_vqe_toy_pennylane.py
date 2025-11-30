#!/usr/bin/env python
"""
Demo 02 - VQE Toy Energy Minimization (PennyLane)

This demo:
- Builds a simple 2-qubit Hamiltonian (toy H2-like)
- Defines a variational ansatz circuit
- Runs a gradient-based VQE optimization
- Plots energy convergence vs. iteration
- Saves results as JSON and PNG
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

import pennylane as qml
from pennylane import numpy as pnp


def build_hamiltonian():
    """Construct a simple 2-qubit Hamiltonian (toy H2 example)."""
    coeffs = [-1.05, -0.39, 0.1]
    ops = [
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.PauliX(0) @ qml.PauliX(1),
    ]
    return qml.Hamiltonian(coeffs, ops)


def ansatz(params, wires):
    """Simple variational ansatz."""
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=wires)


def vqe_run(output_dir, max_steps=80, stepsize=0.4):
    """Run VQE optimization loop."""

    os.makedirs(output_dir, exist_ok=True)

    H = build_hamiltonian()
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def cost(params):
        ansatz(params, wires=[0, 1])
        return qml.expval(H)

    params = pnp.array([0.0, 0.0], requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=stepsize)

    energies = []
    params_history = []

    for step in range(max_steps):
        params, energy = opt.step_and_cost(cost, params)
        energies.append(float(energy))
        params_history.append(params.tolist())

        if step % 10 == 0 or step == max_steps - 1:
            print(f"[step {step}] Energy = {energy:.6f}")

    # Save JSON results
    json_path = os.path.join(output_dir, "results_demo02_vqe.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "energies": energies,
                "params_history": params_history,
                "final_energy": energies[-1],
            },
            f,
            indent=2,
        )

    # Plot convergence
    plt.figure(figsize=(8, 5))
    plt.plot(range(max_steps), energies, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Demo 02 - VQE Energy Convergence")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "energy_convergence_demo02.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved JSON to {json_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results/demo02")
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--stepsize", type=float, default=0.4)
    args = parser.parse_args()

    vqe_run(args.output_dir, args.max_steps, args.stepsize)
