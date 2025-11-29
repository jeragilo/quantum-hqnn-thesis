#!/usr/bin/env python
"""
Demo 01 - HQNN Toy Classifier (Qiskit AerSimulator)

Binary classification using:
- Classical preprocessing
- Simple variational quantum circuit
- AerSimulator (with optional noise model)
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


def build_feature_map(num_qubits, x):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(x[i], i)
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


def create_noise_model(p=0.01):
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(p, 1),
                                      ["rx", "ry", "rz", "u1", "u2", "u3"])
    noise.add_all_qubit_quantum_error(depolarizing_error(p, 2),
                                      ["cx", "cz"])
    return noise


def circuit_expval_from_counts(counts):
    shots = sum(counts.values())
    exp = 0.0
    for bitstring, c in counts.items():
        parity = bitstring.count("1") % 2
        value = 1 if parity == 0 else -1
        exp += (value * c / shots)
    return exp


def predict_probs(simulator, num_qubits, weights, X, shots=1024):
    probs = []
    for x in X:
        x_pad = np.zeros(num_qubits)
        x_pad[: len(x)] = x
        qc = build_hqnn_circuit(num_qubits, x_pad, weights)

        result = simulator.run(qc, shots=shots).result()
        counts = result.get_counts()
        expval = circuit_expval_from_counts(counts)
        p1 = (1 - expval) / 2
        probs.append(p1)
    return np.array(probs)


def accuracy_from_probs(probs, y_true):
    return np.mean((probs >= 0.5).astype(int) == y_true)


def plot_accuracy(history, path):
    steps = [h["step"] for h in history]
    train = [h["train_acc"] for h in history]
    test = [h["test_acc"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, train, marker="o", label="Train Acc")
    plt.plot(steps, test, marker="s", label="Test Acc")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_demo(output_dir, use_noise=False):
    np.random.seed(42)

    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=2,
	n_redundant=0,
	n_repeated=0,
        class_sep=1.5,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    num_qubits = 4
    num_params = 2 * num_qubits

    if use_noise:
        simulator = AerSimulator(noise_model=create_noise_model())
    else:
        simulator = AerSimulator()

    weights = np.random.uniform(-np.pi, np.pi, num_params)
    best_weights = weights.copy()
    best_acc = 0.0
    history = []

    for step in range(50):
        candidate = best_weights + 0.1 * np.random.randn(num_params)
        train_probs = predict_probs(simulator, num_qubits, candidate, X_train)
        train_acc = accuracy_from_probs(train_probs, y_train)

        if train_acc > best_acc:
            best_acc = train_acc
            best_weights = candidate

        if step % 5 == 0 or step == 49:
            test_probs = predict_probs(simulator, num_qubits, best_weights, X_test)
            test_acc = accuracy_from_probs(test_probs, y_test)
            history.append(
                {"step": step, "train_acc": float(best_acc), "test_acc": float(test_acc)}
            )
            print(f"[step {step}] train_acc={best_acc:.3f} | test_acc={test_acc:.3f}")

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "results_demo01.json")
    plot_path = os.path.join(output_dir, "accuracy_demo01.png")

    with open(json_path, "w") as f:
        json.dump({"history": history, "best_acc": float(best_acc)}, f, indent=2)

    plot_accuracy(history, plot_path)

    print(f"Saved {json_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/demo01")
    parser.add_argument("--noise", action="store_true")
    args = parser.parse_args()
    run_demo(args.output_dir, use_noise=args.noise)
