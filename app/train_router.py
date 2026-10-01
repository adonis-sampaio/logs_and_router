import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from typing import Dict, List

from .causal_analyzer import CausalChainAnalyzer
from .knn_router import KNNRouter
from .models_layer import ModelLayer


def load_cases(dataset_path: Path) -> List[Dict]:
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or len(cases) < 10:
        raise ValueError("The dataset must contain at least ten incidents.")
    required_categories = ModelLayer.ROOT_CAUSE_CATEGORIES
    seen_ids = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each dataset item must be a JSON object.")
        incident_id = case.get("id")
        if not incident_id or incident_id in seen_ids:
            raise ValueError("Every incident must have a unique, non-empty id.")
        seen_ids.add(incident_id)
        if case.get("expected_category") not in required_categories:
            raise ValueError(f"Incident {incident_id!r} has an unsupported expected_category.")
        if not isinstance(case.get("logs"), list) or not case["logs"]:
            raise ValueError(f"Incident {incident_id!r} must contain a non-empty logs array.")
    return cases


async def benchmark_cases(
    cases: List[Dict], model_ids: List[str], model_layer: ModelLayer
) -> List[Dict]:
    analyzer = CausalChainAnalyzer()
    examples = []
    for case in cases:
        logs = case["logs"]
        context = case.get("context", "")
        chain = analyzer.analyze(logs)
        outcomes = {}

        started = time.perf_counter()
        rule_category = (chain.get("root_cause") or {}).get("category", "unknown")
        outcomes["rule_based"] = {
            "quality": float(rule_category == case["expected_category"]),
            "cost_usd": 0.0,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }

        for model_id in model_ids:
            result = await asyncio.to_thread(
                model_layer.openrouter_diagnosis, model_id, logs, chain
            )
            outcomes[f"openrouter:{model_id}"] = {
                "quality": float(result["category"] == case["expected_category"]),
                "cost_usd": result["cost_usd"],
                "latency_ms": result["latency_ms"],
            }

        examples.append(
            {
                "id": case["id"],
                "logs": logs,
                "context": context,
                "expected_category": case["expected_category"],
                "outcomes": outcomes,
            }
        )
    return examples


def summarize_outcomes(examples: List[Dict], backend: str) -> Dict:
    results = [example["outcomes"][backend] for example in examples]
    return {
        "accuracy": sum(result["quality"] for result in results) / len(results),
        "mean_cost_usd": sum(result["cost_usd"] for result in results) / len(results),
        "mean_latency_ms": sum(result["latency_ms"] for result in results) / len(results),
    }


def evaluate_policy(
    examples: List[Dict],
    artifact_path: Path,
    min_quality: float,
    max_cost_usd: float,
    test_fraction: float,
    seed: int,
) -> Dict:
    if len(examples) < 10:
        raise ValueError("At least ten distinct incidents are required for held-out evaluation.")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    test_size = max(1, min(len(shuffled) - 2, round(len(shuffled) * test_fraction)))
    test_examples = shuffled[:test_size]
    train_examples = shuffled[test_size:]

    router = KNNRouter(
        model_path=str(artifact_path),
        min_quality=min_quality,
        max_cost_usd=max_cost_usd,
    )
    router.train(train_examples)
    available_backends = sorted(examples[0]["outcomes"])

    selected_results = []
    for example in test_examples:
        decision = router.predict(
            example["logs"],
            example.get("context", ""),
            available_backends=available_backends,
        )
        actual = example["outcomes"][decision["backend"]]
        selected_results.append(actual)

    routed = {
        "accuracy": sum(result["quality"] for result in selected_results) / test_size,
        "mean_cost_usd": sum(result["cost_usd"] for result in selected_results) / test_size,
        "mean_latency_ms": sum(result["latency_ms"] for result in selected_results) / test_size,
        "quality_target": min_quality,
        "max_cost_usd": max_cost_usd,
    }
    baselines = {
        backend: summarize_outcomes(test_examples, backend)
        for backend in available_backends
    }
    training_baselines = {
        backend: summarize_outcomes(train_examples, backend)
        for backend in available_backends
    }
    qualifying_training_baselines = [
        (backend, summary)
        for backend, summary in training_baselines.items()
        if summary["accuracy"] >= min_quality
        and summary["mean_cost_usd"] <= max_cost_usd
    ]
    selected_fixed_backend = (
        min(qualifying_training_baselines, key=lambda item: item[1]["mean_cost_usd"])[0]
        if qualifying_training_baselines
        else "rule_based"
    )
    report = {
        "protocol": "random incident-level holdout; exact root-cause category accuracy",
        "seed": seed,
        "training_incidents": len(train_examples),
        "test_incidents": test_size,
        "router": routed,
        "fixed_backend_baselines": baselines,
        "fixed_backend_selected_on_training_data": selected_fixed_backend,
        "selected_fixed_backend_test_results": baselines[selected_fixed_backend],
        "artifact": str(artifact_path),
        "note": "Results are only as representative as the supplied labeled incident dataset.",
    }
    router.train(examples, min_quality=min_quality, max_cost_usd=max_cost_usd)
    return report


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure analysis backends and evaluate quality/cost-aware KNN routing."
    )
    parser.add_argument("dataset", type=Path, help="Labeled incident dataset in JSON format")
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated OpenRouter model IDs to benchmark alongside rule_based",
    )
    parser.add_argument("--outcomes", type=Path, default=Path("models/routing-outcomes.json"))
    parser.add_argument("--artifact", type=Path, default=Path("models/knn_router.pkl"))
    parser.add_argument("--min-quality", type=float, default=0.8)
    parser.add_argument("--max-cost-usd", type=float, default=0.01)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    model_ids = [model_id.strip() for model_id in args.models.split(",") if model_id.strip()]
    model_layer = ModelLayer()
    if model_ids and not model_layer.openrouter_configured:
        raise RuntimeError("OPENROUTER_API_KEY is required to benchmark OpenRouter models.")
    measured = await benchmark_cases(cases, model_ids, model_layer)
    args.outcomes.parent.mkdir(parents=True, exist_ok=True)
    args.outcomes.write_text(json.dumps(measured, indent=2), encoding="utf-8")

    report = evaluate_policy(
        measured,
        args.artifact,
        min_quality=args.min_quality,
        max_cost_usd=args.max_cost_usd,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    report["outcomes_file"] = str(args.outcomes)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())