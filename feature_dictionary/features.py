"""
Core feature data structures.

Represents features extracted from SAE and their activations.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import math


class FeatureType(Enum):
    """Types of features based on activation patterns."""
    SPARSE = "sparse"           # Activates rarely
    MODERATE = "moderate"       # Moderate activation frequency
    DENSE = "dense"             # Activates frequently
    DEAD = "dead"               # Never activates
    ULTRA_SPARSE = "ultra_sparse"  # Extremely rare activation


@dataclass
class FeatureActivation:
    """A single activation of a feature."""
    feature_id: int
    token: str
    position: int
    activation: float
    context: str = ""
    context_tokens: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "token": self.token,
            "position": self.position,
            "activation": self.activation,
            "context": self.context,
            "context_tokens": self.context_tokens,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureActivation":
        return cls(
            feature_id=data["feature_id"],
            token=data["token"],
            position=data["position"],
            activation=data["activation"],
            context=data.get("context", ""),
            context_tokens=data.get("context_tokens", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Feature:
    """A single feature from the SAE."""
    id: int
    dimension: int
    decoder_weight: Optional[List[float]] = None
    encoder_weight: Optional[List[float]] = None
    bias: float = 0.0

    # Activation statistics
    activation_count: int = 0
    mean_activation: float = 0.0
    max_activation: float = 0.0
    activation_frequency: float = 0.0

    # Top activating examples
    top_activations: List[FeatureActivation] = field(default_factory=list)

    # Labels and descriptions
    labels: List[str] = field(default_factory=list)
    description: str = ""
    confidence: float = 0.0

    # Feature type
    feature_type: FeatureType = FeatureType.SPARSE

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_activation(
        self,
        activation: FeatureActivation,
        max_examples: int = 20,
    ) -> None:
        """Add an activation, keeping top N by activation value."""
        self.activation_count += 1

        # Update running mean
        prev_mean = self.mean_activation
        self.mean_activation = prev_mean + (activation.activation - prev_mean) / self.activation_count

        # Update max
        if activation.activation > self.max_activation:
            self.max_activation = activation.activation

        # Add to top activations
        self.top_activations.append(activation)
        self.top_activations.sort(key=lambda a: a.activation, reverse=True)
        self.top_activations = self.top_activations[:max_examples]

    def classify_type(self, total_tokens: int) -> FeatureType:
        """Classify feature type based on activation frequency."""
        if total_tokens == 0:
            self.feature_type = FeatureType.DEAD
        else:
            self.activation_frequency = self.activation_count / total_tokens

            if self.activation_frequency == 0:
                self.feature_type = FeatureType.DEAD
            elif self.activation_frequency < 0.0001:
                self.feature_type = FeatureType.ULTRA_SPARSE
            elif self.activation_frequency < 0.01:
                self.feature_type = FeatureType.SPARSE
            elif self.activation_frequency < 0.1:
                self.feature_type = FeatureType.MODERATE
            else:
                self.feature_type = FeatureType.DENSE

        return self.feature_type

    def get_top_tokens(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N tokens by activation value."""
        return [(a.token, a.activation) for a in self.top_activations[:n]]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dimension": self.dimension,
            "bias": self.bias,
            "activation_count": self.activation_count,
            "mean_activation": self.mean_activation,
            "max_activation": self.max_activation,
            "activation_frequency": self.activation_frequency,
            "top_activations": [a.to_dict() for a in self.top_activations],
            "labels": self.labels,
            "description": self.description,
            "confidence": self.confidence,
            "feature_type": self.feature_type.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Feature":
        feature = cls(
            id=data["id"],
            dimension=data["dimension"],
            bias=data.get("bias", 0.0),
        )
        feature.activation_count = data.get("activation_count", 0)
        feature.mean_activation = data.get("mean_activation", 0.0)
        feature.max_activation = data.get("max_activation", 0.0)
        feature.activation_frequency = data.get("activation_frequency", 0.0)
        feature.top_activations = [
            FeatureActivation.from_dict(a) for a in data.get("top_activations", [])
        ]
        feature.labels = data.get("labels", [])
        feature.description = data.get("description", "")
        feature.confidence = data.get("confidence", 0.0)
        feature.feature_type = FeatureType(data.get("feature_type", "sparse"))
        feature.metadata = data.get("metadata", {})
        return feature


@dataclass
class FeatureSet:
    """Collection of features from an SAE layer."""
    layer: int
    features: Dict[int, Feature] = field(default_factory=dict)
    total_tokens_processed: int = 0
    model_name: str = ""
    sae_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, feature_id: int) -> Feature:
        return self.features[feature_id]

    def __iter__(self):
        return iter(self.features.values())

    def add_feature(self, feature: Feature) -> None:
        """Add a feature to the set."""
        self.features[feature.id] = feature

    def get_feature(self, feature_id: int) -> Optional[Feature]:
        """Get a feature by ID."""
        return self.features.get(feature_id)

    def get_features_by_type(self, feature_type: FeatureType) -> List[Feature]:
        """Get all features of a specific type."""
        return [f for f in self.features.values() if f.feature_type == feature_type]

    def get_features_by_label(self, label: str) -> List[Feature]:
        """Get all features with a specific label."""
        return [f for f in self.features.values() if label in f.labels]

    def classify_all_types(self) -> Dict[FeatureType, int]:
        """Classify all features and return counts by type."""
        counts = {ft: 0 for ft in FeatureType}
        for feature in self.features.values():
            feature.classify_type(self.total_tokens_processed)
            counts[feature.feature_type] += 1
        return counts

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.features:
            return {
                "total_features": 0,
                "active_features": 0,
                "dead_features": 0,
                "labeled_features": 0,
                "mean_activation_frequency": 0.0,
            }

        active = sum(1 for f in self.features.values() if f.activation_count > 0)
        dead = sum(1 for f in self.features.values() if f.feature_type == FeatureType.DEAD)
        labeled = sum(1 for f in self.features.values() if f.labels)

        freqs = [f.activation_frequency for f in self.features.values()]
        mean_freq = sum(freqs) / len(freqs) if freqs else 0.0

        return {
            "total_features": len(self.features),
            "active_features": active,
            "dead_features": dead,
            "labeled_features": labeled,
            "mean_activation_frequency": mean_freq,
            "total_tokens_processed": self.total_tokens_processed,
        }

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "features": {str(k): v.to_dict() for k, v in self.features.items()},
            "total_tokens_processed": self.total_tokens_processed,
            "model_name": self.model_name,
            "sae_name": self.sae_name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSet":
        feature_set = cls(
            layer=data["layer"],
            total_tokens_processed=data.get("total_tokens_processed", 0),
            model_name=data.get("model_name", ""),
            sae_name=data.get("sae_name", ""),
            metadata=data.get("metadata", {}),
        )
        for k, v in data.get("features", {}).items():
            feature_set.features[int(k)] = Feature.from_dict(v)
        return feature_set

    def save(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FeatureSet":
        """Load from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
