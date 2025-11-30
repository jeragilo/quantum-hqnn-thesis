#!/usr/bin/env python
"""
Demo 08 — Hybrid HQNN Training Loop (Qiskit + SPSA)

This demo:
- Uses the HQNN variational circuit (same as Demo 01)
- Builds a hybrid quantum-classical training loop
- Uses SPSA for gradient approximation (robust under noise)
- Trains on a synthetic binary dataset
- Tracks loss + accuracy over epochs
- Saves results as JSON and PNG plots

Outputs:
- results_demo08_training.json
- training_curves_demo08.png
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ============================================================
# HQNN Circuit Components
# ============================================================

def build_feature_map(num_qubits, x):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(float(x[i]), i)
    return qc

def build_variational_layer(num_qubits, weights):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.rx(weights[i], i)
        qc.rz(weights[num_qubits + i], i)
    for i in range(num_qubits):
        qc.cz(i, (i + 1) % num_qubits)
    return qc

def build_hqnn_circuit(num_qubits, x, weights):
    fm = build_feature_map(num_qubits, x)
    var = build_variational_layer(num_qubits, weights)
    qc = fm.compose(var)
    qc.measure_all()
    return qc

# ============================================================
# Parity Expectation
# ============================================================

def parity_expval(counts):
    shots = sum(counts.values())
    exp = 0
    for bitstring, c in counts.items():
        parity = bitstring.count("1") % 2
        val = 1 if parity == 0 else -1
        exp += val * c / shots
    return exp

def predict_prob(sim, num_qubits, weights, x):
    x_pad = np.zeros(num_qubits)
    x_pad[:len(x)] = x
    qc = build_hqnn_circuit(num_qubits, x_pad, weights)
    result = sim.run(qc, shots=1024).result()
    counts = result.get_counts()
    exp = parity_expval(counts)
    p1 = (1 - exp) / 2
    return p1

# ============================================================
# Loss + Accuracy
# ============================================================

def loss_fn(sim, num_qubits, weights, X, y):
    preds = np.array([predict_prob(sim, num_qubits, weights, x) for x in X])
    # Binary cross-entropy
    eps = 1e-10
    loss = -np.mean(y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps))
    return loss

def accuracy(sim, num_qubits, weights, X, y):
    preds = np.array([predict_prob(sim, num_qubits, weights, x) for x in X])
    return np.mean((preds >= 0.5).astype(int) == y)

# ============================================================
# SPSA Optimizer
# ============================================================

def spsa_update(sim, num_qubits, weights, X, y, alpha=0.1, c=0.1):
    dim = len(weights)
    delta = 2 * np.random.randint(0, 2, dim) - 1  # ±1

    w_plus = weights + c * delta
    w_minus = weights - c * delta

    loss_plus = loss_fn(sim, num_qubits, w_plus, X, y)
    loss_minus = loss_fn(sim, num_qubits, w_minus, X, y)

    g_hat = (loss_plus - loss_minus) / (2 * c * delta)
    new_weights = weights - alpha * g_hat
    return new_weights

# ============================================================
# Training Loop
# ============================================================

def run_demo(output_dir, epochs=20):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Dataset
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        class_sep=1.4,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    num_qubits = 4
    num_params = 2 * num_qubits
    weights = np.random.uniform(-np.pi, np.pi, num_params)

    sim = AerSimulator()

    history = {
        "loss": [],
        "accuracy": []
    }

    # 2. Training Loop
    for epoch in range(epochs):
        weights = spsa_update(sim, num_qubits, weights, X_train, y_train)

        loss = loss_fn(sim, num_qubits, weights, X_train, y_train)
        acc = accuracy(sim, num_qubits, weights, X_test, y_test)

        history["loss"].append(float(loss))
        history["accuracy"].append(float(acc))

        print(f"[Epoch {epoch}] Loss={loss:.4f} | Test Accuracy={acc:.4f}")

    # 3. Save JSON
    json_path = os.path.join(output_dir, "results_demo08_training.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)

    # 4. Plot results
    plt.figure(figsize=(10,5))
    plt.plot(history["loss"], label="Loss")
    plt.plot(history["accuracy"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Demo 08 — HQNN Training Curve (SPSA)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "training_curves_demo08.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"\nSaved JSON to {json_path}")
    print(f"Saved plot to {plot_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/demo08")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    run_demo(args.output_dir, epochs=args.epochs)

