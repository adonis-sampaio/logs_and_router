import asyncio

from app.train_router import benchmark_cases
from app.train_router import evaluate_policy
from app.knn_router import KNNRouter


def measured_example(incident_id, message, outcomes):
    return {
        "id": incident_id,
        "logs": [{"message": message, "source": "api", "level": "ERROR"}],
        "context": "incident triage",
        "expected_category": "database",
        "outcomes": outcomes,
    }


def test_benchmark_records_backend_quality_cost_and_latency():
    class MeasuredModelLayer:
        def openrouter_diagnosis(self, model_id, logs, chain):
            return {
                "category": "database",
                "explanation": "measured response",
                "cost_usd": 0.003,
                "latency_ms": 28.0,
            }

    cases = [
        {
            "id": "db-incident",
            "expected_category": "database",
            "logs": [
                {
                    "timestamp": "2026-09-28T10:00:00Z",
                    "source": "api",
                    "level": "ERROR",
                    "message": "database connection timeout",
                }
            ],
        }
    ]

    measured = asyncio.run(
        benchmark_cases(cases, ["provider/model"], MeasuredModelLayer())
    )

    assert measured[0]["outcomes"]["rule_based"]["quality"] == 1.0
    assert measured[0]["outcomes"]["openrouter:provider/model"]["quality"] == 1.0
    assert measured[0]["outcomes"]["openrouter:provider/model"]["cost_usd"] == 0.003
    assert measured[0]["outcomes"]["openrouter:provider/model"]["latency_ms"] == 28.0


def test_untrained_router_uses_rule_based_fallback(tmp_path):
    router = KNNRouter(model_path=str(tmp_path / "missing.pkl"))

    decision = router.predict([], available_backends=["rule_based", "openrouter:model"])

    assert decision["backend"] == "rule_based"
    assert decision["policy_source"] == "untrained_fallback"
    assert decision["estimated_quality"] is None


def test_router_chooses_cheapest_backend_meeting_quality_target(tmp_path):
    router = KNNRouter(
        n_neighbors=2,
        model_path=str(tmp_path / "router.pkl"),
        min_quality=0.8,
        max_cost_usd=0.02,
    )
    outcomes = {
        "rule_based": {"quality": 0.0, "cost_usd": 0.0, "latency_ms": 1.0},
        "openrouter:small": {"quality": 1.0, "cost_usd": 0.005, "latency_ms": 50.0},
        "openrouter:large": {"quality": 1.0, "cost_usd": 0.015, "latency_ms": 100.0},
    }
    examples = [
        measured_example(str(index), f"database failure pattern {index}", outcomes)
        for index in range(3)
    ]
    router.train(examples)

    decision = router.predict(
        examples[0]["logs"],
        examples[0]["context"],
        available_backends=outcomes,
    )

    assert decision["backend"] == "openrouter:small"
    assert decision["selection_reason"] == "lowest_cost_meeting_quality_target"
    assert decision["estimated_quality"] == 1.0
    assert decision["estimated_cost_usd"] == 0.005


def test_router_uses_best_affordable_backend_below_quality_target(tmp_path):
    router = KNNRouter(
        n_neighbors=2,
        model_path=str(tmp_path / "router.pkl"),
        min_quality=0.9,
        max_cost_usd=0.001,
    )
    outcomes = {
        "rule_based": {"quality": 0.0, "cost_usd": 0.0, "latency_ms": 1.0},
        "openrouter:small": {"quality": 1.0, "cost_usd": 0.005, "latency_ms": 50.0},
    }
    examples = [
        measured_example(str(index), f"database failure pattern {index}", outcomes)
        for index in range(3)
    ]
    router.train(examples)

    decision = router.predict(
        examples[0]["logs"],
        examples[0]["context"],
        available_backends=outcomes,
    )

    assert decision["backend"] == "rule_based"
    assert decision["selection_reason"] == "best_quality_within_budget"
    assert decision["estimated_quality"] < decision["quality_target"]


def test_router_falls_back_for_out_of_distribution_features(tmp_path):
    router = KNNRouter(n_neighbors=2, model_path=str(tmp_path / "router.pkl"))
    outcomes = {
        "rule_based": {"quality": 0.5, "cost_usd": 0.0, "latency_ms": 1.0},
        "openrouter:small": {"quality": 1.0, "cost_usd": 0.005, "latency_ms": 50.0},
    }
    examples = [
        measured_example(str(index), f"database error {index}", outcomes)
        for index in range(3)
    ]
    router.train(examples)

    decision = router.predict(
        [{"message": "x" * 100_000, "source": "unseen", "level": "ERROR"}],
        "unseen context " * 1000,
        available_backends=outcomes,
    )

    assert decision["backend"] == "rule_based"
    assert decision["policy_source"] == "out_of_distribution_fallback"
    assert decision["estimated_quality"] is None


def test_policy_evaluation_uses_held_out_incidents_and_reports_baselines(tmp_path):
    examples = []
    for index in range(12):
        category_correct = float(index % 2 == 0)
        outcomes = {
            "rule_based": {
                "quality": category_correct,
                "cost_usd": 0.0,
                "latency_ms": 1.0,
            },
            "openrouter:small": {
                "quality": 1.0,
                "cost_usd": 0.002,
                "latency_ms": 25.0,
            },
        }
        examples.append(measured_example(str(index), f"case {index}", outcomes))

    report = evaluate_policy(
        examples,
        artifact_path=tmp_path / "router.pkl",
        min_quality=0.8,
        max_cost_usd=0.01,
        test_fraction=0.25,
        seed=4,
    )

    assert report["training_incidents"] == 9
    assert report["test_incidents"] == 3
    assert report["router"]["accuracy"] >= 0.0
    assert report["router"]["mean_cost_usd"] >= 0.0
    assert set(report["fixed_backend_baselines"]) == {"rule_based", "openrouter:small"}
    assert report["fixed_backend_selected_on_training_data"] in report["fixed_backend_baselines"]
    deployed_router = KNNRouter(model_path=str(tmp_path / "router.pkl"))
    assert deployed_router.artifact["training_incidents"] == len(examples)