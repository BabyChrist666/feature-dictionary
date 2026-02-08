"""
Feature labeling and description generation.

Uses various strategies to assign semantic labels to features.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from enum import Enum
import re
from collections import Counter

from .features import Feature, FeatureSet


class LabelingStrategy(Enum):
    """Strategies for generating labels."""
    TOKEN_FREQUENCY = "token_frequency"      # Most common tokens
    NGRAM_ANALYSIS = "ngram_analysis"        # N-gram patterns
    PATTERN_MATCHING = "pattern_matching"    # Regex patterns
    SEMANTIC_CLUSTERING = "semantic_clustering"  # Embedding similarity
    MANUAL = "manual"                        # Human-provided


@dataclass
class Label:
    """A semantic label for a feature."""
    text: str
    confidence: float
    strategy: LabelingStrategy
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "strategy": self.strategy.value,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Label":
        return cls(
            text=data["text"],
            confidence=data["confidence"],
            strategy=LabelingStrategy(data["strategy"]),
            evidence=data.get("evidence", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LabelingConfig:
    """Configuration for labeling."""
    # Token frequency settings
    min_token_frequency: int = 3
    max_labels: int = 5

    # Pattern matching
    patterns: Dict[str, str] = field(default_factory=dict)

    # Confidence thresholds
    min_confidence: float = 0.3

    # N-gram settings
    ngram_sizes: List[int] = field(default_factory=lambda: [1, 2, 3])

    # Strategies to use
    strategies: List[LabelingStrategy] = field(
        default_factory=lambda: [
            LabelingStrategy.TOKEN_FREQUENCY,
            LabelingStrategy.NGRAM_ANALYSIS,
        ]
    )

    def to_dict(self) -> dict:
        return {
            "min_token_frequency": self.min_token_frequency,
            "max_labels": self.max_labels,
            "patterns": self.patterns,
            "min_confidence": self.min_confidence,
            "ngram_sizes": self.ngram_sizes,
            "strategies": [s.value for s in self.strategies],
        }


class FeatureLabeler:
    """
    Assigns semantic labels to features.

    Uses multiple strategies to generate meaningful labels
    based on activation patterns.
    """

    def __init__(self, config: Optional[LabelingConfig] = None):
        self.config = config or LabelingConfig()

        # Default patterns for common categories
        self.patterns = {
            "number": r"^\d+$",
            "punctuation": r"^[^\w\s]+$",
            "uppercase": r"^[A-Z]+$",
            "code_keyword": r"^(if|else|for|while|def|class|import|return|function|const|let|var)$",
            "whitespace": r"^\s+$",
            "article": r"^(a|an|the)$",
            "pronoun": r"^(i|you|he|she|it|we|they|me|him|her|us|them)$",
            "preposition": r"^(in|on|at|to|for|with|by|from|of|about)$",
        }
        self.patterns.update(self.config.patterns)

    def label_feature(self, feature: Feature) -> List[Label]:
        """
        Generate labels for a single feature.

        Args:
            feature: Feature to label

        Returns:
            List of labels with confidence scores
        """
        labels = []

        for strategy in self.config.strategies:
            if strategy == LabelingStrategy.TOKEN_FREQUENCY:
                labels.extend(self._label_by_token_frequency(feature))
            elif strategy == LabelingStrategy.NGRAM_ANALYSIS:
                labels.extend(self._label_by_ngrams(feature))
            elif strategy == LabelingStrategy.PATTERN_MATCHING:
                labels.extend(self._label_by_patterns(feature))

        # Filter by confidence and sort
        labels = [l for l in labels if l.confidence >= self.config.min_confidence]
        labels.sort(key=lambda l: l.confidence, reverse=True)

        # Keep top labels
        labels = labels[:self.config.max_labels]

        # Update feature
        feature.labels = [l.text for l in labels]
        if labels:
            feature.confidence = labels[0].confidence

        return labels

    def label_features(self, feature_set: FeatureSet) -> Dict[int, List[Label]]:
        """
        Label all features in a set.

        Args:
            feature_set: Set of features to label

        Returns:
            Dict mapping feature IDs to their labels
        """
        results = {}
        for feature_id, feature in feature_set.features.items():
            results[feature_id] = self.label_feature(feature)
        return results

    def _label_by_token_frequency(self, feature: Feature) -> List[Label]:
        """Generate labels based on most common tokens."""
        labels = []

        if not feature.top_activations:
            return labels

        # Count tokens
        token_counts = Counter()
        for act in feature.top_activations:
            token = act.token.strip().lower()
            if token:
                token_counts[token] += 1

        # Create labels from frequent tokens
        total = len(feature.top_activations)
        for token, count in token_counts.most_common(5):
            if count >= self.config.min_token_frequency:
                confidence = count / total
                labels.append(Label(
                    text=f"token:{token}",
                    confidence=confidence,
                    strategy=LabelingStrategy.TOKEN_FREQUENCY,
                    evidence=[token] * min(count, 3),
                ))

        return labels

    def _label_by_ngrams(self, feature: Feature) -> List[Label]:
        """Generate labels based on n-gram patterns."""
        labels = []

        if not feature.top_activations:
            return labels

        # Collect context tokens
        all_ngrams: Dict[int, Counter] = {n: Counter() for n in self.config.ngram_sizes}

        for act in feature.top_activations:
            tokens = act.context_tokens
            for n in self.config.ngram_sizes:
                for i in range(len(tokens) - n + 1):
                    ngram = " ".join(tokens[i:i+n])
                    all_ngrams[n][ngram] += 1

        # Find significant n-grams
        for n, ngram_counts in all_ngrams.items():
            if not ngram_counts:
                continue

            total = sum(ngram_counts.values())
            for ngram, count in ngram_counts.most_common(3):
                if count >= 2:
                    confidence = count / total
                    if confidence >= 0.2:
                        labels.append(Label(
                            text=f"ngram:{ngram}",
                            confidence=confidence,
                            strategy=LabelingStrategy.NGRAM_ANALYSIS,
                            evidence=[ngram],
                            metadata={"n": n, "count": count},
                        ))

        return labels

    def _label_by_patterns(self, feature: Feature) -> List[Label]:
        """Generate labels based on regex patterns."""
        labels = []

        if not feature.top_activations:
            return labels

        # Check each pattern
        pattern_matches: Dict[str, int] = {}

        for act in feature.top_activations:
            token = act.token
            for pattern_name, pattern in self.patterns.items():
                if re.match(pattern, token, re.IGNORECASE):
                    pattern_matches[pattern_name] = pattern_matches.get(pattern_name, 0) + 1

        # Create labels from matching patterns
        total = len(feature.top_activations)
        for pattern_name, count in pattern_matches.items():
            confidence = count / total
            if confidence >= 0.3:
                labels.append(Label(
                    text=f"pattern:{pattern_name}",
                    confidence=confidence,
                    strategy=LabelingStrategy.PATTERN_MATCHING,
                    evidence=[pattern_name],
                    metadata={"count": count},
                ))

        return labels

    def generate_description(self, feature: Feature) -> str:
        """
        Generate a natural language description for a feature.

        Args:
            feature: Feature to describe

        Returns:
            Description string
        """
        parts = []

        # Feature type
        parts.append(f"A {feature.feature_type.value} feature")

        # Activation stats
        if feature.activation_count > 0:
            parts.append(
                f"that activated {feature.activation_count} times "
                f"(frequency: {feature.activation_frequency:.4f})"
            )

        # Labels
        if feature.labels:
            label_str = ", ".join(feature.labels[:3])
            parts.append(f"characterized by: {label_str}")

        # Top tokens
        top_tokens = feature.get_top_tokens(5)
        if top_tokens:
            token_str = ", ".join([f"'{t[0]}'" for t in top_tokens])
            parts.append(f"Top activating tokens: {token_str}")

        description = ". ".join(parts) + "."
        feature.description = description
        return description

    def add_pattern(self, name: str, pattern: str) -> None:
        """Add a custom pattern for matching."""
        self.patterns[name] = pattern

    def find_similar_features(
        self,
        feature: Feature,
        feature_set: FeatureSet,
        min_overlap: float = 0.3,
    ) -> List[Tuple[int, float]]:
        """
        Find features with similar activation patterns.

        Args:
            feature: Reference feature
            feature_set: Set to search
            min_overlap: Minimum token overlap ratio

        Returns:
            List of (feature_id, similarity) tuples
        """
        similar = []

        if not feature.top_activations:
            return similar

        # Get token set for reference feature
        ref_tokens = set(a.token.lower() for a in feature.top_activations)

        for other_id, other in feature_set.features.items():
            if other_id == feature.id:
                continue

            if not other.top_activations:
                continue

            other_tokens = set(a.token.lower() for a in other.top_activations)

            # Calculate Jaccard similarity
            intersection = len(ref_tokens & other_tokens)
            union = len(ref_tokens | other_tokens)

            if union > 0:
                similarity = intersection / union
                if similarity >= min_overlap:
                    similar.append((other_id, similarity))

        return sorted(similar, key=lambda x: x[1], reverse=True)
