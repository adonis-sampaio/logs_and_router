from typing import Dict, List


class CausalMetrics:
    @staticmethod
    def causal_chain_recall(predicted_chain: Dict, ground_truth: Dict) -> float:
        """Calculate what percentage of the true causal chain was detected."""
        true_nodes = set(n.get("event_id", n.get("eventid")) for n in ground_truth.get("nodes", []))
        predicted_nodes = set(
            n.get("event_id", n.get("eventid")) for n in predicted_chain.get("nodes", [])
        )

        if not true_nodes:
            return 1.0 if not predicted_nodes else 0.0

        intersection = true_nodes.intersection(predicted_nodes)
        return len(intersection) / len(true_nodes)

    @staticmethod
    def causalchainrecall(predicted_chain: Dict, ground_truth: Dict) -> float:
        return CausalMetrics.causal_chain_recall(predicted_chain, ground_truth)

    @staticmethod
    def ordered_causal_chain_recall(predicted_chain: Dict, ground_truth: Dict) -> float:
        """Calculate recall considering the order of the causal chain."""
        if not ground_truth.get("nodes"):
            return 1.0 if not predicted_chain.get("nodes") else 0.0

        true_sequence = sorted(
            ground_truth["nodes"], key=lambda x: x.get("timestamp", "")
        )
        predicted_sequence = sorted(
            predicted_chain["nodes"], key=lambda x: x.get("timestamp", "")
        )

        lcs_length = CausalMetrics.longest_common_subsequence(
            [n.get("event_id", n.get("eventid")) for n in true_sequence],
            [n.get("event_id", n.get("eventid")) for n in predicted_sequence],
        )
        return lcs_length / len(true_sequence)

    @staticmethod
    def orderedcausalchainrecall(predicted_chain: Dict, ground_truth: Dict) -> float:
        return CausalMetrics.ordered_causal_chain_recall(predicted_chain, ground_truth)

    @staticmethod
    def rootcauseaccuracy(predicted_root: Dict, true_root: Dict) -> float:
        """Check if the correct root cause was identified."""
        if predicted_root.get("event_id") == true_root.get("event_id"):
            return 1.0

        if predicted_root.get("category") == true_root.get("category"):
            return 0.5

        return 0.0

    @staticmethod
    def longest_common_subsequence(seq1: List[str], seq2: List[str]) -> int:
        """Find the length of the longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    @staticmethod
    def longestcommonsubsequence(seq1: List[str], seq2: List[str]) -> int:
        return CausalMetrics.longest_common_subsequence(seq1, seq2)
