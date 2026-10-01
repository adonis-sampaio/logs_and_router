from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_infer_root_cause_database():
    response = client.post(
        "/infer-root-cause",
        json={"log_text": "Database connection timeout while querying users"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["likely_root_cause"] == "database"
    assert payload["confidence"] >= 0.5


def test_static_evaluation_summary_endpoint_is_removed():
    response = client.get("/evaluation-summary")
    assert response.status_code == 404


def test_openrouter_models_reports_unconfigured(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    response = client.get("/openrouter/models")

    assert response.status_code == 200
    assert response.json() == {
        "configured": False,
        "default_model": None,
        "models": [],
    }


def test_openrouter_catalog_includes_only_zero_cost_free_models(monkeypatch):
    monkeypatch.setattr(
        main.model_layer,
        "_openrouter_request",
        lambda method, url: {
            "data": [
                {
                    "id": "paid/model:free",
                    "name": "Paid free suffix",
                    "pricing": {"prompt": "0.01", "completion": "0.02"},
                },
                {
                    "id": "provider/paid-model",
                    "name": "Paid model",
                    "pricing": {"prompt": "0", "completion": "0"},
                },
                {
                    "id": "provider/free-model:free",
                    "name": "Free model",
                    "pricing": {"prompt": "0", "completion": "0"},
                },
            ]
        },
    )

    models = main.model_layer.list_free_openrouter_models()

    assert models == [{"id": "provider/free-model:free", "name": "Free model"}]


def test_openrouter_diagnosis_parses_category_and_reported_cost(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        main.model_layer,
        "_openrouter_request",
        lambda method, url, payload=None: {
            "choices": [
                {
                    "message": {
                        "content": '{"category":"network","explanation":"DNS failures are the supporting evidence."}'
                    }
                }
            ],
            "usage": {"cost": 0.0025},
        },
    )

    diagnosis = main.model_layer.openrouter_diagnosis(
        "provider/model",
        [{"message": "DNS lookup failed"}],
        {"nodes": [], "edges": [], "root_cause": {}},
    )

    assert diagnosis["category"] == "network"
    assert diagnosis["explanation"].startswith("DNS failures")
    assert diagnosis["cost_usd"] == 0.0025
    assert diagnosis["latency_ms"] >= 0.0


def test_analyze_uses_selected_free_openrouter_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        main.model_layer,
        "openrouter_diagnosis",
        lambda model_id, logs, chain: {
            "category": "database",
            "explanation": f"Explanation from {model_id}",
            "cost_usd": 0.0,
            "latency_ms": 12.0,
        },
    )

    response = client.post(
        "/analyze",
        json={
            "logs": [
                {
                    "timestamp": "2026-09-28T10:00:00Z",
                    "source": "api",
                    "level": "ERROR",
                    "message": "Database connection timeout",
                }
            ],
            "llm_model": "provider/model:free",
        },
    )

    assert response.status_code == 200
    assert response.json()["model_used"] == "provider/model:free"
    assert response.json()["explanation"] == "Explanation from provider/model:free"
    assert response.json()["root_cause_category"] == "database"
    assert response.json()["routing"]["policy_source"] == "manual_override"
    assert response.json()["routing"]["actual_cost_usd"] == 0.0


def test_analyze_automatic_route_calls_selected_backend(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("ROUTER_CANDIDATE_MODELS", "provider/model:free")
    monkeypatch.setattr(
        main.router,
        "predict",
        lambda logs, context, available_backends: {
            "backend": "openrouter:provider/model:free",
            "policy_source": "knn_measured_outcomes",
            "selection_reason": "lowest_cost_meeting_quality_target",
            "estimated_quality": 0.9,
            "estimated_cost_usd": 0.001,
            "estimated_latency_ms": 20.0,
            "quality_target": 0.8,
            "max_cost_usd": 0.01,
        },
    )
    monkeypatch.setattr(
        main.model_layer,
        "openrouter_diagnosis",
        lambda model_id, logs, chain: {
            "category": "network",
            "explanation": "Provider diagnosis",
            "cost_usd": 0.0,
            "latency_ms": 14.0,
        },
    )

    response = client.post(
        "/analyze",
        json={
            "logs": [
                {
                    "timestamp": "2026-09-28T10:00:00Z",
                    "source": "api",
                    "level": "ERROR",
                    "message": "DNS resolution failure",
                }
            ]
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_used"] == "provider/model:free"
    assert payload["root_cause_category"] == "network"
    assert payload["routing"]["policy_source"] == "knn_measured_outcomes"
    assert payload["routing"]["estimated_cost_usd"] == 0.001
    assert payload["routing"]["actual_cost_usd"] == 0.0
    assert payload["routing"]["actual_backend"] == "openrouter:provider/model:free"


def test_automatic_route_falls_back_to_rules_when_provider_fails(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        main.router,
        "predict",
        lambda logs, context, available_backends: {
            "backend": "openrouter:provider/model:free",
            "policy_source": "knn_measured_outcomes",
            "selection_reason": "lowest_cost_meeting_quality_target",
            "estimated_quality": 0.9,
            "estimated_cost_usd": 0.0,
            "estimated_latency_ms": 20.0,
            "quality_target": 0.8,
            "max_cost_usd": 0.01,
        },
    )

    def fail_provider(model_id, logs, chain):
        raise main.OpenRouterError("provider unavailable")

    monkeypatch.setattr(main.model_layer, "openrouter_diagnosis", fail_provider)
    response = client.post(
        "/analyze",
        json={
            "logs": [
                {
                    "timestamp": "2026-09-28T10:00:00Z",
                    "source": "api",
                    "level": "ERROR",
                    "message": "DNS resolution failure",
                }
            ]
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_used"] == "rule_based"
    assert payload["routing"]["backend"] == "openrouter:provider/model:free"
    assert payload["routing"]["actual_backend"] == "rule_based"
    assert payload["routing"]["execution_fallback"] == "provider_request_failed"


def test_analyze_rejects_paid_openrouter_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    response = client.post(
        "/analyze",
        json={
            "logs": [
                {
                    "timestamp": "2026-09-28T10:00:00Z",
                    "source": "api",
                    "level": "ERROR",
                    "message": "Database connection timeout",
                }
            ],
            "llm_model": "provider/model",
        },
    )

    assert response.status_code == 400
    assert "ending in ':free'" in response.json()["detail"]
