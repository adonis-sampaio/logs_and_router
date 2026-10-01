from datetime import datetime
from typing import Dict, List


class CausalChainAnalyzer:
    """Very small causal-chain analyzer used by the API."""

    def __init__(self):
        self.causal_patterns = {
            "database": [
                "database",
                "query",
                "sql",
                "timeout",
                "connection",
                "deadlock",
                "transaction",
            ],
            "network": [
                "network",
                "dns",
                "latency",
                "timeout",
                "connection refused",
                "socket",
                "packet loss",
            ],
            "memory": ["memory", "oom", "heap", "gc", "leak", "out of memory"],
            "cpu": ["cpu", "thread", "loop", "high load", "processor", "utilization"],
            "disk": ["disk", "i/o", "storage", "full", "no space left", "fsck"],
            "authentication": ["auth", "token", "login", "credential", "forbidden", "401", "403"],
        }
        self.root_cause_hints = {
            "database": "Likely database or query layer issue such as connection exhaustion, timeouts, or bad queries.",
            "network": "Likely connectivity or routing issue such as DNS, firewall, or unstable network connections.",
            "memory": "Likely memory pressure, leak, or runaway allocation causing application instability.",
            "cpu": "Likely CPU saturation or a computational loop causing resource exhaustion.",
            "disk": "Likely storage exhaustion or disk I/O bottleneck causing slowdowns or failures.",
            "authentication": "Likely credential, token, or permission problem affecting access and validation.",
            "unknown": "No strong evidence was found in the symptom text; this may require additional logs or context.",
        }

    def analyze(self, logs: List[Dict]) -> Dict:
        """Build a minimal causal chain from a list of log records."""
        categorized = self.categorize_logs(logs)
        nodes = []
        edges = []

        for i, log in enumerate(categorized):
            node = {
                "event_id": f"event_{i}",
                "timestamp": log["timestamp"],
                "description": log["message"],
                "nodetype": self.determine_node_type(log, categorized),
                "confidence": self.calculate_confidence(log),
                "category": log.get("category", "unknown"),
            }
            nodes.append(node)

        for i in range(len(nodes) - 1):
            if self.has_causal_relationship(nodes[i], nodes[i + 1]):
                edge = {
                    "source": nodes[i]["event_id"],
                    "target": nodes[i + 1]["event_id"],
                    "weight": self.calculate_edge_weight(nodes[i], nodes[i + 1]),
                }
                edges.append(edge)

        root_cause = self.find_root_cause(nodes)
        return {"nodes": nodes, "edges": edges, "root_cause": root_cause}

    def categorize_logs(self, logs: List[Dict]) -> List[Dict]:
        categorized = []
        for log in logs:
            message = (log.get("message", "") or "").lower()
            category = "unknown"
            for key, keywords in self.causal_patterns.items():
                if any(keyword in message for keyword in keywords):
                    category = key
                    break
            categorized.append({**log, "category": category})
        return categorized

    def determine_node_type(self, log: Dict, logs: List[Dict]) -> str:
        category_logs = [item for item in logs if item.get("category") == log.get("category")]
        if not category_logs:
            return "root_cause"

        timestamps = [
            datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
            for item in category_logs
        ]
        earliest = min(timestamps)
        current = datetime.fromisoformat(log["timestamp"].replace("Z", "+00:00"))

        if current == earliest:
            return "root_cause"
        if log.get("level", "").upper() in ["ERROR", "CRITICAL"]:
            return "propagation"
        return "symptom"

    def calculate_confidence(self, log: Dict) -> float:
        confidence = 0.5
        if log.get("level", "").upper() in ["ERROR", "CRITICAL"]:
            confidence += 0.3
        if log.get("category", "unknown") != "unknown":
            confidence += 0.2
        return min(confidence, 1.0)

    def has_causal_relationship(self, node1: Dict, node2: Dict) -> bool:
        if not node1.get("timestamp") or not node2.get("timestamp"):
            return False

        t1 = datetime.fromisoformat(node1["timestamp"].replace("Z", "+00:00"))
        t2 = datetime.fromisoformat(node2["timestamp"].replace("Z", "+00:00"))
        time_diff = abs((t2 - t1).total_seconds())

        if time_diff > 300:
            return False

        category_map = {
            "database": ["network", "memory"],
            "network": ["disk"],
            "memory": ["cpu"],
            "cpu": ["memory"],
        }
        cat1 = node1.get("category", "unknown")
        cat2 = node2.get("category", "unknown")
        if cat2 in category_map.get(cat1, []):
            return True
        return node1.get("nodetype") == "root_cause" or node2.get("nodetype") == "symptom"

    def calculate_edge_weight(self, node1: Dict, node2: Dict) -> float:
        t1 = datetime.fromisoformat(node1["timestamp"].replace("Z", "+00:00"))
        t2 = datetime.fromisoformat(node2["timestamp"].replace("Z", "+00:00"))
        time_diff = abs((t2 - t1).total_seconds())
        return max(0.1, 1.0 - (time_diff / 300.0))

    def find_root_cause(self, nodes: List[Dict]) -> Dict:
        if not nodes:
            return {}
        return max(nodes, key=lambda node: node.get("confidence", 0.0))

    def infer_root_cause_from_symptom(self, logs: List[Dict]) -> Dict:
        """Infer likely root cause from a symptom-only log list."""
        scored_causes = {}
        evidence_by_category = {}

        for log in logs:
            message = (log.get("message", "") or "").lower()
            if not message:
                continue

            for category, keywords in self.causal_patterns.items():
                matched = [keyword for keyword in keywords if keyword in message]
                if matched:
                    scored_causes[category] = scored_causes.get(category, 0.0) + float(len(matched))
                    evidence_by_category.setdefault(category, [])
                    evidence_by_category[category].extend(matched)

        if not scored_causes:
            return {
                "likely_root_cause": "unknown",
                "confidence": 0.0,
                "reason": self.root_cause_hints["unknown"],
                "possible_causes": [
                    {"category": "unknown", "score": 0.0, "reason": self.root_cause_hints["unknown"], "evidence": []}
                ],
                "evidence": [],
                "debug": {},
            }

        ranked = sorted(scored_causes.items(), key=lambda item: item[1], reverse=True)
        top_category, top_score = ranked[0]
        possible_causes = []
        for category, score in ranked[:3]:
            possible_causes.append(
                {
                    "category": category,
                    "score": round(score / top_score, 2) if top_score else 0.0,
                    "reason": self.root_cause_hints.get(category, self.root_cause_hints["unknown"]),
                    "evidence": sorted(set(evidence_by_category.get(category, []))),
                }
            )

        evidence = sorted(set(item for values in evidence_by_category.values() for item in values))
        return {
            "likely_root_cause": top_category,
            "confidence": round(min(0.95, 0.55 + (top_score / max(len(self.causal_patterns[top_category]), 1)) * 0.35), 2),
            "reason": self.root_cause_hints.get(top_category, self.root_cause_hints["unknown"]),
            "possible_causes": possible_causes,
            "evidence": evidence,
            "debug": {category: sorted(set(v)) for category, v in evidence_by_category.items()},
        }

    def explain_inference(self, logs: List[Dict]) -> Dict:
        """Return a debug-friendly explanation of matched symptoms and candidate categories."""
        return self.infer_root_cause_from_symptom(logs)
