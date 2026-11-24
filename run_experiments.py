import json
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    compare_classical_models,
    train_linear_regression,
    train_logistic_classifier,
)

def run_regression_experiment():
    df = pd.read_csv("Student_Performance.csv")
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

    X = df.drop(columns=["Performance Index"]).values
    y = df["Performance Index"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    reg_input = RegressionInput(
        train_features=X_train_scaled,
        train_target=y_train,
        test_features=X_test_scaled,
        test_target=y_test,
        learning_rate=0.05,
        epochs=800,
    )
    res = train_linear_regression(reg_input)
    test_mse = float(np.mean((res.predictions - y_test) ** 2))

    print("=== Линейная регрессия (Student_Performance.csv) ===")
    print(f"Начальная MSE (train): {res.loss_history[0]:.4f}")
    print(f"Финальная MSE (train): {res.loss_history[-1]:.4f}")
    print(f"Итоговая MSE (test):   {test_mse:.4f}")
    print(f"Веса: {np.round(res.weights, 4)}, смещение: {res.bias:.4f}\n")

    return {"test_mse": test_mse}

def run_classification_experiment():
    df = pd.read_csv("german.csv", sep=";")

    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    cls_input = ClassificationInput(
        train_features=X_train_scaled,
        train_target=y_train,
        test_features=X_test_scaled,
        test_target=y_test,
        learning_rate=0.1,
        epochs=1000,
    )

    own_res = train_logistic_classifier(cls_input)
    comp_res = compare_classical_models(cls_input)

    print("=== Классификация (german.csv) ===")
    print(f"Начальный LogLoss (train): {own_res.loss_history[0]:.4f}")
    print(f"Финальный LogLoss (train): {own_res.loss_history[-1]:.4f}")
    print("Своя логистическая регрессия:", comp_res.logistic_regression)
    print("Decision Tree:               ", comp_res.decision_tree)
    print("kNN:                         ", comp_res.knn)

    return {
        "own_logistic": asdict(comp_res.logistic_regression),
        "logistic_regression": asdict(comp_res.logistic_regression),
        "decision_tree": asdict(comp_res.decision_tree),
        "knn": asdict(comp_res.knn),
    }

if __name__ == "__main__":
    results = {
        "linear_regression": run_regression_experiment(),
        "classification": run_classification_experiment(),
    }

    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nГотово! Результаты сохранены в experiment_results.json")
