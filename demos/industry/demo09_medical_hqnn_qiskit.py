#!/usr/bin/env python
"""
Demo 09 — Medical Risk Classification (HQNN vs Classical Logistic Regression)

This industry demo:
- Creates a small synthetic "medical risk" dataset (4 features, binary label).
- Trains a hybrid HQNN classifier (Qiskit + SPSA) using the same architecture as Demo 01/08.
- Trains a classical LogisticRegression baseline.
- Compares test accuracy between HQNN and classical model.
- Saves JSON and a bar chart for later thesis integration.

Outputs:
- results/demo09/results_demo09_medical.json
- results/demo09/accuracy_demo09.png
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ============================================================
# HQNN Circuit Components (same style as Demo 01 / 08)
# ============================================================

def build_feature_map(num_qubits, x):
    """Simple RY feature encoding."""
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(float(x[i]), i)
    return qc


def build_variational_layer(num_qubits, weights):
    """RX + RZ + CZ entangling block."""
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.rx(weights[i], i)
        qc.rz(weights[num_qubits + i], i)
    for i in range(num_qubits):
        qc.cz(i, (i + 1) % num_qubits)
    return qc


def build_hqnn_circuit(num_qubits, x, weights):
    """Full HQNN circuit with measurement."""
    fm = build_feature_map(num_qubits, x)
    var = build_variational_layer(num_qubits, weights)
    qc = fm.compose(var)
    qc.measure_all()
    return qc


# ============================================================
# Parity → Probability
# ============================================================

def parity_expval(counts):
    shots = sum(counts.values())
    exp = 0.0
    for bitstring, c in counts.items():
        parity = bitstring.count("1") % 2
        sign = 1 if parity == 0 else -1
        exp += sign * c / shots
    return exp


def predict_prob(sim, num_qubits, weights, x):
    """Run HQNN and return P(y=1) from parity expectation."""
    x_pad = np.zeros(num_qubits)
    x_pad[:len(x)] = x

    qc = build_hqnn_circuit(num_qubits, x_pad, weights)
    result = sim.run(qc, shots=1024).result()
    counts = result.get_counts()
    exp = parity_expval(counts)
    p1 = (1 - exp) / 2.0
    return p1


# ============================================================
# Loss, Accuracy, SPSA
# ============================================================

def loss_fn(sim, num_qubits, weights, X, y):
    preds = np.array([predict_prob(sim, num_qubits, weights, x) for x in X])
    eps = 1e-10
    # Binary cross-entropy
    loss = -np.mean(y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps))
    return loss


def accuracy_hqnn(sim, num_qubits, weights, X, y):
    preds = np.array([predict_prob(sim, num_qubits, weights, x) for x in X])
    y_hat = (preds >= 0.5).astype(int)
    return float(np.mean(y_hat == y))


def spsa_update(sim, num_qubits, weights, X, y, alpha=0.1, c=0.1):
    dim = len(weights)
    delta = 2 * np.random.randint(0, 2, dim) - 1  # ±1 perturbation vector

    w_plus = weights + c * delta
    w_minus = weights - c * delta

    loss_plus = loss_fn(sim, num_qubits, w_plus, X, y)
    loss_minus = loss_fn(sim, num_qubits, w_minus, X, y)

    g_hat = (loss_plus - loss_minus) / (2 * c * delta)
    new_weights = weights - alpha * g_hat
    return new_weights


# ============================================================
# Main Demo
# ============================================================

def run_demo(output_dir, epochs=12):
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # 1. Synthetic "medical risk" dataset
    # --------------------------------------------------------
    # 4 features ~ imagine: age, blood pressure, cholesterol, heart rate
    X, y = make_classification(
        n_samples=250,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        class_sep=1.6,
        random_state=7,
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

    history = {"loss": [], "hqnn_accuracy": []}

    # --------------------------------------------------------
    # 2. HQNN Training loop (SPSA)
    # --------------------------------------------------------
    for epoch in range(epochs):
        weights = spsa_update(sim, num_qubits, weights, X_train, y_train, alpha=0.15, c=0.15)

        loss = loss_fn(sim, num_qubits, weights, X_train, y_train)
        acc = accuracy_hqnn(sim, num_qubits, weights, X_test, y_test)

        history["loss"].append(float(loss))
        history["hqnn_accuracy"].append(float(acc))

        print(f"[HQNN][Epoch {epoch}] Loss={loss:.4f} | Test Accuracy={acc:.4f}")

    final_hqnn_acc = history["hqnn_accuracy"][-1]

    # --------------------------------------------------------
    # 3. Classical Logistic Regression baseline
    # --------------------------------------------------------
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred_cl = clf.predict(X_test)
    classical_acc = float(accuracy_score(y_test, y_pred_cl))

    # --------------------------------------------------------
    # 4. Save JSON summary
    # --------------------------------------------------------
    summary = {
        "hqnn_final_accuracy": final_hqnn_acc,
        "classical_accuracy": classical_acc,
        "epochs": epochs,
        "hqnn_loss_history": history["loss"],
        "hqnn_accuracy_history": history["hqnn_accuracy"],
    }

    json_path = os.path.join(output_dir, "results_demo09_medical.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # --------------------------------------------------------
    # 5. Plot accuracy comparison
    # --------------------------------------------------------
    labels = ["HQNN (Hybrid)", "Logistic Regression"]
    accs = [final_hqnn_acc, classical_acc]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, accs, color=["purple", "gray"])
    plt.ylabel("Test Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("Demo 09 — Medical Risk Classification: HQNN vs Classical")
    plt.tight_layout()

    png_path = os.path.join(output_dir, "accuracy_demo09.png")
    plt.savefig(png_path)
    plt.close()

    print("\n===== DEMO 09 SUMMARY =====")
    print(summary)
    print(f"\nSaved JSON to: {json_path}")
    print(f"Saved plot to: {png_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/demo09")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    run_demo(args.output_dir, epochs=args.epochs)
