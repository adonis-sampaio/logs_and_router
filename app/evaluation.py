import argparse
import hashlib
import json
import math
import platform
import random
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import sklearn
from sklearn.model_selection import train_test_split

from .causal_analyzer import CausalChainAnalyzer


EVALUATOR_VERSION = "1.0"
MIN_BOOTSTRAP_SAMPLES = 100


def load_dataset(dataset_path: Path) -> Dict:
    """Load a versioned incident dataset and require its privacy-review attestation."""
    try:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read a valid JSON dataset: {exc}") from exc

    if not isinstance(dataset, dict) or dataset.get("schema_version") != 1:
        raise ValueError("Dataset must be an object with schema_version 1.")
    for field in ("dataset_id", "dataset_version", "labeling_guidelines"):
        if not isinstance(dataset.get(field), str) or not dataset[field].strip():
            raise ValueError(f"Dataset field {field!r} must be a non-empty string.")

    privacy_review = dataset.get("privacy_review")
    if not isinstance(privacy_review, dict) or privacy_review.get("status") != "approved":
        raise ValueError("Dataset privacy_review.status must be 'approved'.")
    try:
        date.fromisoformat(privacy_review.get("reviewed_on", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError("privacy_review.reviewed_on must be an ISO date (YYYY-MM-DD).") from exc
    if not isinstance(privacy_review.get("redaction_method"), str) or not privacy_review[
        "redaction_method"
    ].strip():
        raise ValueError("privacy_review.redaction_method must describe the redaction process.")

    incidents = dataset.get("incidents")
    if not isinstance(incidents, list) or not incidents:
        raise ValueError("Dataset incidents must be a non-empty array.")

    analyzer = CausalChainAnalyzer()
    categories = set(analyzer.causal_patterns) | {"unknown"}
    seen_ids = set()
    for incident in incidents:
        if not isinstance(incident, dict):
            raise ValueError("Every incident must be a JSON object.")
        incident_id = incident.get("id")
        if not isinstance(incident_id, str) or not incident_id.strip() or incident_id in seen_ids:
            raise ValueError("Every incident must have a unique, non-empty string id.")
        seen_ids.add(incident_id)
        expected_category = incident.get("expected_category")
        if not isinstance(expected_category, str) or expected_category not in categories:
            raise ValueError(f"Incident {incident_id!r} has an unsupported expected_category.")
        logs = incident.get("logs")
        if not isinstance(logs, list) or not logs:
            raise ValueError(f"Incident {incident_id!r} must contain a non-empty logs array.")
        for log in logs:
            if not isinstance(log, dict):
                raise ValueError(f"Incident {incident_id!r} contains a non-object log entry.")
            for field in ("timestamp", "source", "level", "message"):
                if not isinstance(log.get(field), str) or not log[field].strip():
                    raise ValueError(
                        f"Incident {incident_id!r} log field {field!r} must be a non-empty string."
                    )
            try:
                datetime.fromisoformat(log["timestamp"].replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(
                    f"Incident {incident_id!r} has a non-ISO timestamp {log['timestamp']!r}."
                ) from exc
    return dataset


def _category_metrics(
    expected: Sequence[str], predicted: Sequence[str], categories: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    results = {}
    for category in categories:
        true_positive = sum(
            actual == category and guess == category
            for actual, guess in zip(expected, predicted)
        )
        false_positive = sum(
            actual != category and guess == category
            for actual, guess in zip(expected, predicted)
        )
        false_negative = sum(
            actual == category and guess != category
            for actual, guess in zip(expected, predicted)
        )
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results[category] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(actual == category for actual in expected),
        }
    return results


def _percentile_interval(values: Sequence[float]) -> List[float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return [round(percentile(0.025), 4), round(percentile(0.975), 4)]


def _summarize_predictions(
    expected: Sequence[str],
    predicted: Sequence[str],
    categories: Sequence[str],
    bootstrap_indices: Sequence[Sequence[int]],
) -> Dict:
    point_metrics = _category_metrics(expected, predicted, categories)
    accuracy = sum(actual == guess for actual, guess in zip(expected, predicted)) / len(expected)
    macro_f1 = sum(item["f1"] for item in point_metrics.values()) / len(point_metrics)
    bootstrap_accuracy = []
    bootstrap_macro_f1 = []
    bootstrap_category_metrics = {
        category: {metric: [] for metric in ("precision", "recall", "f1")}
        for category in categories
    }

    for indices in bootstrap_indices:
        sampled_expected = [expected[index] for index in indices]
        sampled_predicted = [predicted[index] for index in indices]
        sampled_metrics = _category_metrics(sampled_expected, sampled_predicted, categories)
        bootstrap_accuracy.append(
            sum(actual == guess for actual, guess in zip(sampled_expected, sampled_predicted))
            / len(indices)
        )
        bootstrap_macro_f1.append(
            sum(item["f1"] for item in sampled_metrics.values()) / len(sampled_metrics)
        )
        for category in categories:
            for metric in bootstrap_category_metrics[category]:
                bootstrap_category_metrics[category][metric].append(
                    sampled_metrics[category][metric]
                )

    per_category = {}
    for category, metrics in point_metrics.items():
        per_category[category] = {
            "precision": round(metrics["precision"], 4),
            "precision_95ci": _percentile_interval(
                bootstrap_category_metrics[category]["precision"]
            ),
            "recall": round(metrics["recall"], 4),
            "recall_95ci": _percentile_interval(
                bootstrap_category_metrics[category]["recall"]
            ),
            "f1": round(metrics["f1"], 4),
            "f1_95ci": _percentile_interval(bootstrap_category_metrics[category]["f1"]),
            "support": metrics["support"],
        }
    return {
        "test_incidents": len(expected),
        "accuracy": round(accuracy, 4),
        "accuracy_95ci": _percentile_interval(bootstrap_accuracy),
        "macro_f1": round(macro_f1, 4),
        "macro_f1_95ci": _percentile_interval(bootstrap_macro_f1),
        "per_category": per_category,
    }


def evaluate_dataset(
    dataset: Dict,
    test_fraction: float = 0.25,
    seed: int = 42,
    bootstrap_samples: int = 2000,
    dataset_sha256: Optional[str] = None,
) -> Dict:
    """Compare deterministic diagnosis baselines on a stratified incident holdout."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")
    if bootstrap_samples < MIN_BOOTSTRAP_SAMPLES:
        raise ValueError(f"bootstrap_samples must be at least {MIN_BOOTSTRAP_SAMPLES}.")

    incidents = dataset["incidents"]
    labels = [incident["expected_category"] for incident in incidents]
    label_counts = Counter(labels)
    categories = sorted(label_counts)
    test_size = math.ceil(len(incidents) * test_fraction)
    if any(count < 2 for count in label_counts.values()):
        raise ValueError("Stratified holdout requires at least two incidents per category.")
    if test_size < len(categories) or len(incidents) - test_size < len(categories):
        raise ValueError(
            "Holdout size must leave at least one training and one test incident per category; "
            "adjust test_fraction or add incidents."
        )

    all_indices = list(range(len(incidents)))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels,
    )
    training_labels = [labels[index] for index in train_indices]
    expected = [labels[index] for index in test_indices]
    analyzer = CausalChainAnalyzer()
    training_counts = Counter(training_labels)
    majority_category = min(
        training_counts,
        key=lambda category: (-training_counts[category], category),
    )
    predictions = {
        "majority_class": [majority_category] * len(test_indices),
        "symptom_keyword": [],
        "structured_rule": [],
    }
    for index in test_indices:
        logs = incidents[index]["logs"]
        symptom_result = analyzer.infer_root_cause_from_symptom(logs)
        chain_result = analyzer.analyze(logs)
        chain_root = chain_result.get("root_cause") or {}
        predictions["symptom_keyword"].append(symptom_result["likely_root_cause"])
        predictions["structured_rule"].append(chain_root.get("category", "unknown"))

    bootstrap_rng = random.Random(seed)
    bootstrap_indices = [
        [bootstrap_rng.randrange(len(test_indices)) for _ in test_indices]
        for _ in range(bootstrap_samples)
    ]
    results = {
        name: _summarize_predictions(expected, guesses, categories, bootstrap_indices)
        for name, guesses in predictions.items()
    }
    test_counts = Counter(expected)
    report = {
        "evaluator_version": EVALUATOR_VERSION,
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
        "dataset": {
            "id": dataset["dataset_id"],
            "version": dataset["dataset_version"],
            "sha256": dataset_sha256,
            "privacy_review": {
                "status": dataset["privacy_review"]["status"],
                "reviewed_on": dataset["privacy_review"]["reviewed_on"],
            },
        },
        "protocol": {
            "split_unit": "incident",
            "split_strategy": "stratified holdout; all logs for an incident stay together",
            "test_fraction": test_fraction,
            "seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "confidence_level": 0.95,
            "interval_method": "paired percentile bootstrap over held-out incidents",
            "interval_caveat": (
                "Intervals describe sampling uncertainty on this holdout; they do not measure "
                "dataset shift, label ambiguity, or privacy-review completeness."
            ),
        },
        "sample_sizes": {
            "total_incidents": len(incidents),
            "training_incidents": len(train_indices),
            "test_incidents": len(test_indices),
            "per_category": {
                category: {
                    "total": label_counts[category],
                    "training": training_counts[category],
                    "test": test_counts[category],
                }
                for category in categories
            },
        },
        "baselines": {
            "majority_class": {
                "description": "Most frequent training-set category; alphabetical tie-break.",
                "predicted_category": majority_category,
            },
            "symptom_keyword": {
                "description": "Existing symptom keyword scorer, applied to each incident's logs."
            },
            "structured_rule": {
                "description": "Existing structured-log heuristic; evaluates its selected root node category."
            },
        },
        "results": results,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate local incident diagnosis baselines on a privacy-reviewed dataset."
    )
    parser.add_argument("dataset", type=Path, help="Versioned, privacy-reviewed JSON dataset")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("-"), help="Report path, or '-' for stdout")
    args = parser.parse_args()

    try:
        dataset = load_dataset(args.dataset)
        digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
        report = evaluate_dataset(
            dataset,
            test_fraction=args.test_fraction,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
            dataset_sha256=digest,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    output = json.dumps(report, indent=2, sort_keys=True)
    if str(args.out) == "-":
        print(output)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
        print(f"Evaluation report written to {args.out}")


if __name__ == "__main__":
    main()
