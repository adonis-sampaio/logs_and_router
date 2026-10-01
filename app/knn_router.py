import os
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class KNNRouter:
    """Choose the lowest-cost backend predicted to meet a quality target."""

    ARTIFACT_VERSION = 2
    DEFAULT_BACKEND = "rule_based"

    def __init__(
        self,
        n_neighbors: int = 5,
        model_path: Optional[str] = None,
        min_quality: Optional[float] = None,
        max_cost_usd: Optional[float] = None,
    ):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        self.n_neighbors = n_neighbors
        self.model_path = Path(model_path or os.path.join("models", "knn_router.pkl"))
        self.min_quality = (
            float(os.getenv("ROUTER_MIN_QUALITY", "0.8"))
            if min_quality is None
            else min_quality
        )
        self.max_cost_usd = (
            float(os.getenv("ROUTER_MAX_COST_USD", "0.01"))
            if max_cost_usd is None
            else max_cost_usd
        )
        self._validate_constraints(self.min_quality, self.max_cost_usd)
        self.artifact: Optional[Dict] = self._load_artifact()

    @staticmethod
    def _validate_constraints(min_quality: float, max_cost_usd: float) -> None:
        if not math.isfinite(min_quality) or not 0.0 <= min_quality <= 1.0:
            raise ValueError("min_quality must be a finite value between 0 and 1.")
        if not math.isfinite(max_cost_usd) or max_cost_usd < 0.0:
            raise ValueError("max_cost_usd must be a finite non-negative value.")

    @staticmethod
    def extract_features(logs: List[Dict], context: str = "") -> np.ndarray:
        total_length = sum(len(log.get("message", "")) for log in logs)
        unique_sources = len({log.get("source", "") for log in logs})
        error_count = sum(
            1 for log in logs if log.get("level", "").upper() in {"ERROR", "CRITICAL"}
        )
        average_length = total_length / len(logs) if logs else 0.0
        messages = [log.get("message", "") for log in logs]
        unique_message_ratio = len(set(messages)) / len(messages) if messages else 0.0

        return np.array(
            [[
                len(logs),
                total_length,
                unique_sources,
                error_count,
                average_length,
                unique_message_ratio,
                len(context),
                len(context.split()),
            ]],
            dtype=float,
        )

    def train(
        self,
        examples: List[Dict],
        min_quality: Optional[float] = None,
        max_cost_usd: Optional[float] = None,
    ) -> None:
        """Fit from per-incident, measured outcomes for every candidate backend."""
        quality_target = self.min_quality if min_quality is None else min_quality
        cost_limit = self.max_cost_usd if max_cost_usd is None else max_cost_usd
        self._validate_constraints(quality_target, cost_limit)
        if len(examples) < 2:
            raise ValueError("At least two measured incidents are required to train routing.")

        backends = sorted(examples[0].get("outcomes", {}))
        if self.DEFAULT_BACKEND not in backends or not backends:
            raise ValueError("Benchmark outcomes must include the rule_based backend.")

        feature_rows = []
        target_rows = []
        for example in examples:
            outcomes = example.get("outcomes", {})
            if sorted(outcomes) != backends:
                raise ValueError("Every benchmark incident must measure the same backends.")
            feature_rows.append(
                self.extract_features(example.get("logs", []), example.get("context", ""))[0]
            )
            targets = []
            for backend in backends:
                outcome = outcomes[backend]
                quality = float(outcome["quality"])
                cost = float(outcome["cost_usd"])
                latency = float(outcome["latency_ms"])
                if (
                    not math.isfinite(quality)
                    or not 0.0 <= quality <= 1.0
                    or not math.isfinite(cost)
                    or cost < 0.0
                    or not math.isfinite(latency)
                    or latency < 0.0
                ):
                    raise ValueError(f"Invalid measured outcome for backend {backend!r}.")
                targets.extend([quality, cost, latency])
            target_rows.append(targets)

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(np.asarray(feature_rows, dtype=float))
        neighbors = NearestNeighbors(n_neighbors=2).fit(scaled_features)
        training_distances = neighbors.kneighbors(scaled_features)[0][:, 1]
        regressor = KNeighborsRegressor(
            n_neighbors=min(self.n_neighbors, len(examples)), weights="distance"
        )
        regressor.fit(scaled_features, np.asarray(target_rows, dtype=float))

        self.artifact = {
            "version": self.ARTIFACT_VERSION,
            "backends": backends,
            "scaler": scaler,
            "regressor": regressor,
            "max_neighbor_distance": float(np.quantile(training_distances, 0.95)) * 2.0,
            "min_quality": quality_target,
            "max_cost_usd": cost_limit,
            "training_incidents": len(examples),
        }
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.artifact, self.model_path)

    def predict(
        self,
        logs: List[Dict],
        context: str = "",
        available_backends: Optional[Sequence[str]] = None,
        min_quality: Optional[float] = None,
        max_cost_usd: Optional[float] = None,
    ) -> Dict:
        """Return a backend and the measured-outcome estimates behind that choice."""
        artifact = self.artifact
        if not artifact:
            return {
                "backend": self.DEFAULT_BACKEND,
                "policy_source": "untrained_fallback",
                "estimated_quality": None,
                "estimated_cost_usd": None,
                "estimated_latency_ms": None,
                "quality_target": self.min_quality,
                "max_cost_usd": self.max_cost_usd,
            }

        quality_target = artifact["min_quality"] if min_quality is None else min_quality
        cost_limit = artifact["max_cost_usd"] if max_cost_usd is None else max_cost_usd
        self._validate_constraints(quality_target, cost_limit)

        allowed = set(artifact["backends"])
        if available_backends is not None:
            allowed.intersection_update(available_backends)
        if self.DEFAULT_BACKEND not in allowed:
            allowed.add(self.DEFAULT_BACKEND)

        features = self.extract_features(logs, context)
        scaled = artifact["scaler"].transform(features)
        nearest_distance = float(artifact["regressor"].kneighbors(scaled, n_neighbors=1)[0][0, 0])
        max_distance = artifact.get("max_neighbor_distance")
        if max_distance is not None and nearest_distance > max_distance:
            return {
                "backend": self.DEFAULT_BACKEND,
                "policy_source": "out_of_distribution_fallback",
                "selection_reason": "outside_measured_feature_neighborhood",
                "estimated_quality": None,
                "estimated_cost_usd": None,
                "estimated_latency_ms": None,
                "quality_target": quality_target,
                "max_cost_usd": cost_limit,
            }

        predictions = artifact["regressor"].predict(scaled)[0]
        estimates = {}
        for index, backend in enumerate(artifact["backends"]):
            if backend not in allowed:
                continue
            estimates[backend] = {
                "quality": float(np.clip(predictions[index * 3], 0.0, 1.0)),
                "cost_usd": max(0.0, float(predictions[index * 3 + 1])),
                "latency_ms": max(0.0, float(predictions[index * 3 + 2])),
            }

        within_budget = [
            (name, estimate)
            for name, estimate in estimates.items()
            if estimate["cost_usd"] <= cost_limit
        ]
        qualifying = [
            item for item in within_budget if item[1]["quality"] >= quality_target
        ]
        if qualifying:
            backend, selected = min(
                qualifying,
                key=lambda item: (item[1]["cost_usd"], item[1]["latency_ms"], item[0]),
            )
            selection_reason = "lowest_cost_meeting_quality_target"
        elif within_budget:
            backend, selected = max(
                within_budget,
                key=lambda item: (
                    item[1]["quality"],
                    -item[1]["cost_usd"],
                    -item[1]["latency_ms"],
                ),
            )
            selection_reason = "best_quality_within_budget"
        else:
            backend = self.DEFAULT_BACKEND
            selected = estimates.get(backend, {"quality": 0.0, "cost_usd": 0.0, "latency_ms": 0.0})
            selection_reason = "budget_fallback"

        return {
            "backend": backend,
            "policy_source": "knn_measured_outcomes",
            "selection_reason": selection_reason,
            "estimated_quality": round(selected["quality"], 4),
            "estimated_cost_usd": round(selected["cost_usd"], 8),
            "estimated_latency_ms": round(selected["latency_ms"], 2),
            "quality_target": quality_target,
            "max_cost_usd": cost_limit,
        }

    def _load_artifact(self) -> Optional[Dict]:
        if not self.model_path.exists():
            return None
        try:
            artifact = joblib.load(self.model_path)
        except (OSError, ValueError, EOFError):
            return None
        if not isinstance(artifact, dict) or artifact.get("version") != self.ARTIFACT_VERSION:
            return None
        return artifact