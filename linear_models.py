import numpy as np
from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float)
    X_test = np.asarray(data.test_features, dtype=float)

    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0
    loss_history = []

    for _ in range(data.epochs):
        y_pred = X_train @ weights + bias
        error = y_pred - y_train

        mse = float(np.mean(error ** 2))
        loss_history.append(mse)

        dw = (2.0 / n_samples) * (X_train.T @ error)
        db = (2.0 / n_samples) * np.sum(error)

        weights = weights - data.learning_rate * dw
        bias = bias - data.learning_rate * db

    predictions = X_test @ weights + bias

    return RegressionResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        predictions=predictions,
    )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))

def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float)
    X_test = np.asarray(data.test_features, dtype=float)

    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0
    loss_history = []
    eps = 1e-15

    for _ in range(data.epochs):
        linear_output = X_train @ weights + bias
        y_pred = _sigmoid(linear_output)

        p_clipped = np.clip(y_pred, eps, 1.0 - eps)
        loss = -float(
            np.mean(y_train * np.log(p_clipped) + (1.0 - y_train) * np.log(1.0 - p_clipped))
        )
        loss_history.append(loss)

        error = y_pred - y_train
        dw = (1.0 / n_samples) * (X_train.T @ error)
        db = (1.0 / n_samples) * np.sum(error)

        weights = weights - data.learning_rate * dw
        bias = bias - data.learning_rate * db

    test_linear = X_test @ weights + bias
    probabilities = _sigmoid(test_linear)
    predictions = (probabilities >= 0.5).astype(int)

    return ClassificationResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        probabilities=probabilities,
        predictions=predictions,
    )


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> ClassificationMetrics:
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )

def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните вашу логистическую регрессию, Decision Tree и kNN на одном split."""
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=int)
    X_test = np.asarray(data.test_features, dtype=float)
    y_test = np.asarray(data.test_target, dtype=int)

    own_log_res = train_logistic_classifier(data)
    log_metrics = _compute_metrics(
        y_test,
        own_log_res.predictions,
        own_log_res.probabilities,
    )

    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_train, y_train)
    tree_pred = tree_model.predict(X_test)
    tree_proba = tree_model.predict_proba(X_test)[:, 1]
    tree_metrics = _compute_metrics(y_test, tree_pred, tree_proba)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    knn_proba = knn_model.predict_proba(X_test)[:, 1]
    knn_metrics = _compute_metrics(y_test, knn_pred, knn_proba)

    return BaselineComparisonResult(
        logistic_regression=log_metrics,
        decision_tree=tree_metrics,
        knn=knn_metrics,
    )
