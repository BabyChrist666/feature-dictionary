"""
Feature Dictionary - High-level API.

Combines extraction, labeling, and analysis into a unified interface.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterator
from pathlib import Path

from .features import Feature, FeatureSet, FeatureActivation, FeatureType
from .extraction import FeatureExtractor, ExtractionConfig
from .labeling import FeatureLabeler, LabelingConfig, Label
from .analysis import FeatureAnalyzer, ActivationStats


@dataclass
class DictionaryConfig:
    """Configuration for feature dictionary."""
    # Extraction settings
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)

    # Labeling settings
    labeling: LabelingConfig = field(default_factory=LabelingConfig)

    # Dictionary settings
    name: str = "feature_dictionary"
    model_name: str = ""
    sae_name: str = ""
    layer: int = 0

    # Storage
    auto_save: bool = False
    save_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "extraction": self.extraction.to_dict(),
            "labeling": self.labeling.to_dict(),
            "name": self.name,
            "model_name": self.model_name,
            "sae_name": self.sae_name,
            "layer": self.layer,
            "auto_save": self.auto_save,
            "save_path": self.save_path,
        }


class FeatureDictionary:
    """
    High-level interface for feature extraction and analysis.

    Combines:
    - Feature extraction from SAE activations
    - Automatic labeling and description
    - Statistical analysis
    - Search and query capabilities
    """

    def __init__(
        self,
        num_features: int,
        feature_dim: int,
        config: Optional[DictionaryConfig] = None,
    ):
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.config = config or DictionaryConfig()

        # Initialize components
        self.extractor = FeatureExtractor(
            num_features=num_features,
            feature_dim=feature_dim,
            layer=self.config.layer,
            config=self.config.extraction,
        )
        self.labeler = FeatureLabeler(config=self.config.labeling)
        self._analyzer: Optional[FeatureAnalyzer] = None

        # State
        self._finalized = False

    @property
    def features(self) -> FeatureSet:
        """Get the feature set."""
        return self.extractor.features

    @property
    def analyzer(self) -> FeatureAnalyzer:
        """Get the analyzer (creates if needed)."""
        if self._analyzer is None:
            self._analyzer = FeatureAnalyzer(self.features)
        return self._analyzer

    def process(
        self,
        activations: List[List[float]],
        tokens: List[str],
        context: str = "",
    ) -> int:
        """
        Process SAE activations and collect feature data.

        Args:
            activations: SAE feature activations [seq_len, num_features]
            tokens: Token strings for each position
            context: Full text context

        Returns:
            Number of activations recorded
        """
        return self.extractor.process_activations(activations, tokens, context)

    def process_batch(
        self,
        batch_activations: List[List[List[float]]],
        batch_tokens: List[List[str]],
        batch_contexts: Optional[List[str]] = None,
    ) -> int:
        """Process multiple sequences."""
        return self.extractor.process_batch(
            batch_activations, batch_tokens, batch_contexts
        )

    def finalize(self) -> None:
        """
        Finalize processing and compute all statistics.

        Call this after processing all data.
        """
        # Finalize extraction
        self.extractor.finalize()

        # Label all features
        self.labeler.label_features(self.features)

        # Generate descriptions
        for feature in self.features:
            self.labeler.generate_description(feature)

        # Mark as finalized
        self._finalized = True
        self._analyzer = FeatureAnalyzer(self.features)

        # Auto-save if configured
        if self.config.auto_save and self.config.save_path:
            self.save(self.config.save_path)

    def get_feature(self, feature_id: int) -> Optional[Feature]:
        """Get a specific feature."""
        return self.features.get_feature(feature_id)

    def search_by_token(
        self,
        token: str,
        min_activation: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """
        Find features that activate for a token.

        Args:
            token: Token to search for
            min_activation: Minimum activation threshold

        Returns:
            List of (feature_id, max_activation) tuples
        """
        return self.extractor.get_feature_by_token(token, min_activation)

    def search_by_label(self, label: str) -> List[Feature]:
        """
        Find features with a specific label.

        Args:
            label: Label to search for (partial match)

        Returns:
            List of matching features
        """
        results = []
        label_lower = label.lower()
        for feature in self.features:
            for feat_label in feature.labels:
                if label_lower in feat_label.lower():
                    results.append(feature)
                    break
        return results

    def search_by_pattern(self, pattern: str) -> List[Feature]:
        """
        Find features whose top tokens match a pattern.

        Args:
            pattern: Regex pattern to match

        Returns:
            List of matching features
        """
        import re
        results = []
        regex = re.compile(pattern, re.IGNORECASE)

        for feature in self.features:
            for act in feature.top_activations:
                if regex.search(act.token):
                    results.append(feature)
                    break

        return results

    def get_top_features(
        self,
        n: int = 10,
        by: str = "activation_count",
    ) -> List[Feature]:
        """
        Get top N features by a metric.

        Args:
            n: Number of features to return
            by: Metric to sort by (activation_count, max_activation, frequency)

        Returns:
            List of top features
        """
        if by == "activation_count":
            key = lambda f: f.activation_count
        elif by == "max_activation":
            key = lambda f: f.max_activation
        elif by == "frequency":
            key = lambda f: f.activation_frequency
        else:
            key = lambda f: f.activation_count

        sorted_features = sorted(self.features, key=key, reverse=True)
        return sorted_features[:n]

    def get_features_by_type(self, feature_type: FeatureType) -> List[Feature]:
        """Get all features of a specific type."""
        return self.features.get_features_by_type(feature_type)

    def find_similar(
        self,
        feature_id: int,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Find features similar to a given feature.

        Args:
            feature_id: Reference feature
            top_k: Number of similar features to return

        Returns:
            List of (feature_id, similarity) tuples
        """
        feature = self.get_feature(feature_id)
        if not feature:
            return []

        similar = self.labeler.find_similar_features(
            feature, self.features, min_overlap=0.1
        )
        return similar[:top_k]

    def get_correlated(
        self,
        feature_id: int,
        min_correlation: float = 0.3,
    ) -> List[Tuple[int, float]]:
        """Find features that correlate with a given feature."""
        return self.analyzer.find_correlated_features(feature_id, min_correlation)

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return self.analyzer.get_summary()

    def get_feature_stats(self, feature_id: int) -> ActivationStats:
        """Get detailed stats for a feature."""
        feature = self.get_feature(feature_id)
        if not feature:
            return ActivationStats(feature_id=feature_id)
        return self.analyzer.compute_activation_stats(feature)

    def export_feature(self, feature_id: int) -> Dict[str, Any]:
        """Export a single feature as dict."""
        feature = self.get_feature(feature_id)
        if not feature:
            return {"error": "Feature not found"}

        stats = self.get_feature_stats(feature_id)

        return {
            "feature": feature.to_dict(),
            "stats": stats.to_dict(),
        }

    def export_all(self) -> Dict[str, Any]:
        """Export entire dictionary."""
        return {
            "config": self.config.to_dict(),
            "feature_set": self.features.to_dict(),
            "statistics": self.get_statistics(),
        }

    def save(self, path: str) -> None:
        """Save dictionary to file."""
        data = self.export_all()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FeatureDictionary":
        """Load dictionary from file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        feature_set = FeatureSet.from_dict(data["feature_set"])
        num_features = len(feature_set.features)

        # Determine feature dimension from first feature
        feature_dim = 0
        for feature in feature_set.features.values():
            feature_dim = feature.dimension
            break

        dictionary = cls(
            num_features=num_features,
            feature_dim=feature_dim,
        )
        dictionary.extractor.feature_set = feature_set
        dictionary._finalized = True
        dictionary._analyzer = FeatureAnalyzer(feature_set)

        return dictionary

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, feature_id: int) -> Feature:
        feature = self.get_feature(feature_id)
        if feature is None:
            raise KeyError(f"Feature {feature_id} not found")
        return feature

    def __iter__(self) -> Iterator[Feature]:
        return iter(self.features)


def create_dictionary(
    num_features: int,
    feature_dim: int,
    model_name: str = "",
    sae_name: str = "",
    layer: int = 0,
) -> FeatureDictionary:
    """
    Create a feature dictionary with common settings.

    Args:
        num_features: Number of SAE features
        feature_dim: Feature dimensionality
        model_name: Name of the base model
        sae_name: Name of the SAE
        layer: Layer index

    Returns:
        Configured FeatureDictionary
    """
    config = DictionaryConfig(
        model_name=model_name,
        sae_name=sae_name,
        layer=layer,
    )
    return FeatureDictionary(num_features, feature_dim, config)
