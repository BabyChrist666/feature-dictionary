"""
Feature extraction from SAE activations.

Extracts features and collects activation statistics.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterator
import math

from .features import Feature, FeatureSet, FeatureActivation


@dataclass
class ExtractionConfig:
    """Configuration for feature extraction."""
    # Activation thresholds
    activation_threshold: float = 0.0
    min_activation: float = 0.01

    # Example collection
    max_examples_per_feature: int = 20
    context_window: int = 5

    # Batch processing
    batch_size: int = 32

    # Feature filtering
    min_activation_count: int = 1
    include_dead_features: bool = False

    # Metadata
    collect_context: bool = True
    collect_positions: bool = True

    def to_dict(self) -> dict:
        return {
            "activation_threshold": self.activation_threshold,
            "min_activation": self.min_activation,
            "max_examples_per_feature": self.max_examples_per_feature,
            "context_window": self.context_window,
            "batch_size": self.batch_size,
            "min_activation_count": self.min_activation_count,
            "include_dead_features": self.include_dead_features,
            "collect_context": self.collect_context,
            "collect_positions": self.collect_positions,
        }


class FeatureExtractor:
    """
    Extracts features from SAE activations.

    Processes model activations through an SAE and collects
    statistics about which features activate and when.
    """

    def __init__(
        self,
        num_features: int,
        feature_dim: int,
        layer: int = 0,
        config: Optional[ExtractionConfig] = None,
    ):
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.layer = layer
        self.config = config or ExtractionConfig()

        # Initialize feature set
        self.feature_set = FeatureSet(layer=layer)
        for i in range(num_features):
            self.feature_set.add_feature(Feature(id=i, dimension=feature_dim))

        # Processing state
        self._total_tokens = 0
        self._callbacks: List[Callable] = []

    @property
    def features(self) -> FeatureSet:
        """Get the extracted feature set."""
        return self.feature_set

    def add_callback(self, callback: Callable[[FeatureActivation], None]) -> None:
        """Add a callback for each activation."""
        self._callbacks.append(callback)

    def process_activations(
        self,
        activations: List[List[float]],
        tokens: List[str],
        context: str = "",
    ) -> int:
        """
        Process a batch of SAE activations.

        Args:
            activations: SAE feature activations [seq_len, num_features]
            tokens: Token strings for each position
            context: Full text context

        Returns:
            Number of activations recorded
        """
        num_recorded = 0
        seq_len = len(activations)

        for pos, (token, acts) in enumerate(zip(tokens, activations)):
            self._total_tokens += 1

            for feature_id, activation in enumerate(acts):
                if activation >= self.config.min_activation:
                    # Create activation record
                    context_tokens = self._get_context_tokens(
                        tokens, pos, self.config.context_window
                    )

                    act_record = FeatureActivation(
                        feature_id=feature_id,
                        token=token,
                        position=pos,
                        activation=activation,
                        context=context if self.config.collect_context else "",
                        context_tokens=context_tokens,
                    )

                    # Add to feature
                    feature = self.feature_set.features[feature_id]
                    feature.add_activation(
                        act_record,
                        max_examples=self.config.max_examples_per_feature,
                    )

                    # Callbacks
                    for callback in self._callbacks:
                        callback(act_record)

                    num_recorded += 1

        return num_recorded

    def process_batch(
        self,
        batch_activations: List[List[List[float]]],
        batch_tokens: List[List[str]],
        batch_contexts: Optional[List[str]] = None,
    ) -> int:
        """
        Process multiple sequences.

        Args:
            batch_activations: List of [seq_len, num_features] activations
            batch_tokens: List of token lists
            batch_contexts: Optional list of context strings

        Returns:
            Total number of activations recorded
        """
        total = 0
        contexts = batch_contexts or [""] * len(batch_activations)

        for acts, tokens, context in zip(batch_activations, batch_tokens, contexts):
            total += self.process_activations(acts, tokens, context)

        return total

    def _get_context_tokens(
        self,
        tokens: List[str],
        position: int,
        window: int,
    ) -> List[str]:
        """Get surrounding tokens for context."""
        start = max(0, position - window)
        end = min(len(tokens), position + window + 1)
        return tokens[start:end]

    def finalize(self) -> FeatureSet:
        """
        Finalize extraction and compute statistics.

        Returns:
            Completed FeatureSet with all statistics
        """
        self.feature_set.total_tokens_processed = self._total_tokens
        self.feature_set.classify_all_types()

        # Filter dead features if configured
        if not self.config.include_dead_features:
            to_remove = [
                fid for fid, f in self.feature_set.features.items()
                if f.activation_count < self.config.min_activation_count
            ]
            for fid in to_remove:
                del self.feature_set.features[fid]

        return self.feature_set

    def get_top_features(self, n: int = 10) -> List[Feature]:
        """Get top N features by activation count."""
        features = sorted(
            self.feature_set.features.values(),
            key=lambda f: f.activation_count,
            reverse=True,
        )
        return features[:n]

    def get_feature_by_token(
        self,
        token: str,
        min_activation: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """Find features that activate for a specific token."""
        results = []
        for feature in self.feature_set.features.values():
            for act in feature.top_activations:
                if act.token == token and act.activation >= min_activation:
                    results.append((feature.id, act.activation))
                    break
        return sorted(results, key=lambda x: x[1], reverse=True)

    def stream_activations(
        self,
        activation_stream: Iterator[Tuple[List[List[float]], List[str]]],
    ) -> Iterator[int]:
        """
        Process a stream of activations.

        Args:
            activation_stream: Iterator of (activations, tokens) tuples

        Yields:
            Number of activations recorded for each item
        """
        for activations, tokens in activation_stream:
            yield self.process_activations(activations, tokens)


class MockSAE:
    """Mock SAE for testing without real model."""

    def __init__(
        self,
        input_dim: int,
        num_features: int,
        sparsity: float = 0.1,
    ):
        self.input_dim = input_dim
        self.num_features = num_features
        self.sparsity = sparsity

    def encode(self, x: List[List[float]]) -> List[List[float]]:
        """
        Mock encoding with sparse activations.

        Args:
            x: Input activations [seq_len, input_dim]

        Returns:
            SAE activations [seq_len, num_features]
        """
        import random

        seq_len = len(x)
        result = []

        for pos in range(seq_len):
            acts = [0.0] * self.num_features
            # Randomly activate some features
            num_active = max(1, int(self.num_features * self.sparsity))
            for fid in random.sample(range(self.num_features), num_active):
                acts[fid] = random.uniform(0.1, 2.0)
            result.append(acts)

        return result
