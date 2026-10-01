import json

import pytest

from app.evaluation import evaluate_dataset, load_dataset


def make_dataset():
    examples = {
        "database": "database connection pool exhausted",
        "network": "DNS resolution failed",
        "memory": "out of memory in application heap",
        "cpu": "CPU utilization reached high load",
    }
    incidents = []
    for category, message in examples.items():
        for index in range(4):
            incidents.append(
                {
                    "id": f"incident-{category}-{index}",
                    "expected_category": category,
                    "logs": [
                        {
                            "timestamp": f"2026-09-{index + 1:02d}T10:00:00Z",
                            "source": "service",
                            "level": "ERROR",
                            "message": message,
                        }
                    ],
                }
            )
    return {
        "schema_version": 1,
        "dataset_id": "sanitized-test-incidents",
        "dataset_version": "1.0.0",
        "privacy_review": {
            "status": "approved",
            "reviewed_on": "2026-09-30",
            "redaction_method": "Synthetic test data with no user content.",
        },
        "labeling_guidelines": "Each incident is assigned one reviewed root-cause category.",
        "incidents": incidents,
    }


def test_evaluation_is_repeatable_and_reports_holdout_metrics():
    dataset = make_dataset()

    first = evaluate_dataset(
        dataset, test_fraction=0.5, seed=17, bootstrap_samples=100, dataset_sha256="test-hash"
    )
    second = evaluate_dataset(
        dataset, test_fraction=0.5, seed=17, bootstrap_samples=100, dataset_sha256="test-hash"
    )

    assert first == second
    assert first["sample_sizes"]["total_incidents"] == 16
    assert first["sample_sizes"]["training_incidents"] == 8
    assert first["sample_sizes"]["test_incidents"] == 8
    assert set(first["results"]) == {"majority_class", "symptom_keyword", "structured_rule"}
    for result in first["results"].values():
        assert result["test_incidents"] == 8
        assert 0.0 <= result["accuracy"] <= 1.0
        assert len(result["accuracy_95ci"]) == 2
        assert set(result["per_category"]) == {"cpu", "database", "memory", "network"}
        assert all("f1_95ci" in metrics for metrics in result["per_category"].values())
    assert all(
        counts["training"] + counts["test"] == counts["total"]
        for counts in first["sample_sizes"]["per_category"].values()
    )
    assert "incident-database-0" not in json.dumps(first)


def test_load_dataset_requires_privacy_review_attestation(tmp_path):
    dataset = make_dataset()
    dataset_path = tmp_path / "incidents.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    assert load_dataset(dataset_path)["dataset_id"] == "sanitized-test-incidents"

    dataset["privacy_review"]["status"] = "pending"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    with pytest.raises(ValueError, match="must be 'approved'"):
        load_dataset(dataset_path)


def test_evaluation_rejects_categories_too_small_for_stratified_holdout():
    dataset = make_dataset()
    dataset["incidents"] = dataset["incidents"][:5]

    with pytest.raises(ValueError, match="at least two incidents per category"):
        evaluate_dataset(dataset, test_fraction=0.5, bootstrap_samples=100)