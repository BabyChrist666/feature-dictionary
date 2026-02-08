"""
Feature analysis and statistics.

Provides tools for analyzing feature activation patterns.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import Counter
import math

from .features import Feature, FeatureSet, FeatureType


@dataclass
class ActivationStats:
    """Statistics for feature activations."""
    feature_id: int
    activation_count: int = 0
    mean_activation: float = 0.0
    std_activation: float = 0.0
    max_activation: float = 0.0
    min_activation: float = float('inf')
    activation_frequency: float = 0.0

    # Position statistics
    mean_position: float = 0.0
    position_variance: float = 0.0

    # Token diversity
    unique_tokens: int = 0
    token_entropy: float = 0.0

    def to_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "activation_count": self.activation_count,
            "mean_activation": self.mean_activation,
            "std_activation": self.std_activation,
            "max_activation": self.max_activation,
            "min_activation": self.min_activation if self.min_activation != float('inf') else 0.0,
            "activation_frequency": self.activation_frequency,
            "mean_position": self.mean_position,
            "position_variance": self.position_variance,
            "unique_tokens": self.unique_tokens,
            "token_entropy": self.token_entropy,
        }


class FeatureAnalyzer:
    """
    Analyzes feature activation patterns.

    Computes statistics, correlations, and patterns.
    """

    def __init__(self, feature_set: FeatureSet):
        self.feature_set = feature_set

    def compute_activation_stats(self, feature: Feature) -> ActivationStats:
        """
        Compute detailed activation statistics for a feature.

        Args:
            feature: Feature to analyze

        Returns:
            ActivationStats with computed values
        """
        stats = ActivationStats(feature_id=feature.id)

        if not feature.top_activations:
            return stats

        activations = [a.activation for a in feature.top_activations]
        positions = [a.position for a in feature.top_activations]
        tokens = [a.token for a in feature.top_activations]

        # Basic activation stats
        stats.activation_count = feature.activation_count
        stats.mean_activation = sum(activations) / len(activations)
        stats.max_activation = max(activations)
        stats.min_activation = min(activations)
        stats.activation_frequency = feature.activation_frequency

        # Standard deviation
        if len(activations) > 1:
            variance = sum((a - stats.mean_activation) ** 2 for a in activations) / len(activations)
            stats.std_activation = math.sqrt(variance)

        # Position statistics
        if positions:
            stats.mean_position = sum(positions) / len(positions)
            if len(positions) > 1:
                stats.position_variance = sum((p - stats.mean_position) ** 2 for p in positions) / len(positions)

        # Token diversity
        unique = set(tokens)
        stats.unique_tokens = len(unique)

        # Token entropy
        token_counts = Counter(tokens)
        total = len(tokens)
        if total > 0:
            probs = [count / total for count in token_counts.values()]
            stats.token_entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        return stats

    def compute_all_stats(self) -> Dict[int, ActivationStats]:
        """Compute stats for all features."""
        return {
            fid: self.compute_activation_stats(feature)
            for fid, feature in self.feature_set.features.items()
        }

    def find_correlated_features(
        self,
        feature_id: int,
        min_correlation: float = 0.5,
    ) -> List[Tuple[int, float]]:
        """
        Find features that tend to activate together.

        Args:
            feature_id: Reference feature
            min_correlation: Minimum correlation threshold

        Returns:
            List of (feature_id, correlation) tuples
        """
        ref_feature = self.feature_set.features.get(feature_id)
        if not ref_feature or not ref_feature.top_activations:
            return []

        # Get positions where reference activates
        ref_positions = set((a.position, a.context) for a in ref_feature.top_activations)

        correlations = []
        for other_id, other in self.feature_set.features.items():
            if other_id == feature_id:
                continue

            if not other.top_activations:
                continue

            # Get positions for other feature
            other_positions = set((a.position, a.context) for a in other.top_activations)

            # Calculate position overlap
            overlap = len(ref_positions & other_positions)
            union = len(ref_positions | other_positions)

            if union > 0:
                correlation = overlap / union
                if correlation >= min_correlation:
                    correlations.append((other_id, correlation))

        return sorted(correlations, key=lambda x: x[1], reverse=True)

    def find_polysemantic_features(
        self,
        min_clusters: int = 2,
        min_cluster_size: int = 3,
    ) -> List[Tuple[int, List[Set[str]]]]:
        """
        Find features that may be polysemantic.

        Identifies features that activate for distinct token clusters.

        Args:
            min_clusters: Minimum number of distinct clusters
            min_cluster_size: Minimum tokens per cluster

        Returns:
            List of (feature_id, clusters) tuples
        """
        polysemantic = []

        for feature_id, feature in self.feature_set.features.items():
            if not feature.top_activations:
                continue

            # Group tokens by similarity (simple approach: first character)
            clusters: Dict[str, Set[str]] = {}
            for act in feature.top_activations:
                token = act.token.strip().lower()
                if not token:
                    continue

                # Simple clustering by first character
                key = token[0] if token else ""
                if key not in clusters:
                    clusters[key] = set()
                clusters[key].add(token)

            # Filter small clusters
            significant_clusters = [
                c for c in clusters.values()
                if len(c) >= min_cluster_size
            ]

            if len(significant_clusters) >= min_clusters:
                polysemantic.append((feature_id, significant_clusters))

        return polysemantic

    def get_dead_features(self) -> List[int]:
        """Get IDs of features that never activated."""
        return [
            fid for fid, f in self.feature_set.features.items()
            if f.activation_count == 0
        ]

    def get_ultra_sparse_features(
        self,
        max_frequency: float = 0.0001,
    ) -> List[int]:
        """Get IDs of ultra-sparse features."""
        return [
            fid for fid, f in self.feature_set.features.items()
            if 0 < f.activation_frequency < max_frequency
        ]

    def get_dense_features(
        self,
        min_frequency: float = 0.1,
    ) -> List[int]:
        """Get IDs of dense (frequently activating) features."""
        return [
            fid for fid, f in self.feature_set.features.items()
            if f.activation_frequency >= min_frequency
        ]

    def compute_feature_importance(self) -> List[Tuple[int, float]]:
        """
        Compute importance scores for all features.

        Uses a combination of activation frequency and max activation.

        Returns:
            Sorted list of (feature_id, importance) tuples
        """
        importance = []

        for fid, feature in self.feature_set.features.items():
            if feature.activation_count == 0:
                score = 0.0
            else:
                # Combine frequency and max activation
                freq_score = min(feature.activation_frequency * 100, 1.0)
                act_score = min(feature.max_activation / 5.0, 1.0)
                score = 0.7 * freq_score + 0.3 * act_score

            importance.append((fid, score))

        return sorted(importance, key=lambda x: x[1], reverse=True)

    def get_position_distribution(
        self,
        feature_id: int,
    ) -> Dict[str, int]:
        """
        Get position distribution for a feature.

        Args:
            feature_id: Feature to analyze

        Returns:
            Dict with position ranges and counts
        """
        feature = self.feature_set.features.get(feature_id)
        if not feature or not feature.top_activations:
            return {}

        positions = [a.position for a in feature.top_activations]

        return {
            "start": sum(1 for p in positions if p < 5),
            "early": sum(1 for p in positions if 5 <= p < 20),
            "middle": sum(1 for p in positions if 20 <= p < 50),
            "late": sum(1 for p in positions if p >= 50),
        }

    def compare_features(
        self,
        feature_id_1: int,
        feature_id_2: int,
    ) -> Dict[str, Any]:
        """
        Compare two features.

        Args:
            feature_id_1: First feature
            feature_id_2: Second feature

        Returns:
            Comparison results
        """
        f1 = self.feature_set.features.get(feature_id_1)
        f2 = self.feature_set.features.get(feature_id_2)

        if not f1 or not f2:
            return {"error": "Feature not found"}

        # Token overlap
        tokens1 = set(a.token.lower() for a in f1.top_activations)
        tokens2 = set(a.token.lower() for a in f2.top_activations)

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        jaccard = len(intersection) / len(union) if union else 0

        return {
            "feature_1": feature_id_1,
            "feature_2": feature_id_2,
            "activation_count_1": f1.activation_count,
            "activation_count_2": f2.activation_count,
            "mean_activation_1": f1.mean_activation,
            "mean_activation_2": f2.mean_activation,
            "token_overlap": list(intersection)[:10],
            "jaccard_similarity": jaccard,
            "shared_labels": list(set(f1.labels) & set(f2.labels)),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the feature set."""
        stats = self.feature_set.get_statistics()

        # Type distribution
        type_counts = self.feature_set.classify_all_types()

        # Add type distribution to stats
        stats["type_distribution"] = {
            ft.value: count for ft, count in type_counts.items()
        }

        # Compute importance ranking
        importance = self.compute_feature_importance()
        stats["top_features"] = importance[:10]

        return stats
