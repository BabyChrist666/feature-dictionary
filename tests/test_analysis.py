"""Tests for feature analysis."""

import pytest
import math

from feature_dictionary.analysis import (
    ActivationStats,
    FeatureAnalyzer,
)
from feature_dictionary.features import Feature, FeatureSet, FeatureActivation, FeatureType


class TestActivationStats:
    """Tests for ActivationStats."""

    def test_create_stats(self):
        stats = ActivationStats(feature_id=42)
        assert stats.feature_id == 42
        assert stats.activation_count == 0
        assert stats.mean_activation == 0.0

    def test_stats_values(self):
        stats = ActivationStats(
            feature_id=0,
            activation_count=100,
            mean_activation=1.5,
            std_activation=0.5,
            max_activation=3.0,
            unique_tokens=50,
            token_entropy=2.5,
        )
        assert stats.activation_count == 100
        assert stats.mean_activation == 1.5
        assert stats.std_activation == 0.5
        assert stats.token_entropy == 2.5

    def test_to_dict(self):
        stats = ActivationStats(
            feature_id=1,
            activation_count=10,
            mean_activation=1.0,
        )
        d = stats.to_dict()
        assert d["feature_id"] == 1
        assert d["activation_count"] == 10
        assert "token_entropy" in d


class TestFeatureAnalyzer:
    """Tests for FeatureAnalyzer."""

    def create_feature_with_activations(
        self,
        feature_id: int,
        tokens: list,
        activations: list = None,
    ) -> Feature:
        """Helper to create a feature with activations."""
        feature = Feature(id=feature_id, dimension=768)
        if activations is None:
            activations = [1.0] * len(tokens)

        for i, (token, act_val) in enumerate(zip(tokens, activations)):
            act = FeatureActivation(
                feature_id=feature_id,
                token=token,
                position=i,
                activation=act_val,
                context="test context",
            )
            feature.add_activation(act)
        return feature

    def test_create_analyzer(self):
        fs = FeatureSet(layer=0)
        analyzer = FeatureAnalyzer(fs)
        assert analyzer.feature_set is fs

    def test_compute_activation_stats(self):
        fs = FeatureSet(layer=0)
        feature = self.create_feature_with_activations(
            0, ["a", "b", "c"], [1.0, 2.0, 3.0]
        )
        fs.add_feature(feature)

        analyzer = FeatureAnalyzer(fs)
        stats = analyzer.compute_activation_stats(feature)

        assert stats.feature_id == 0
        assert stats.activation_count == 3
        assert stats.mean_activation == 2.0
        assert stats.max_activation == 3.0
        assert stats.min_activation == 1.0

    def test_compute_std_activation(self):
        fs = FeatureSet(layer=0)
        feature = self.create_feature_with_activations(
            0, ["a", "b", "c", "d"], [1.0, 2.0, 3.0, 4.0]
        )
        fs.add_feature(feature)

        analyzer = FeatureAnalyzer(fs)
        stats = analyzer.compute_activation_stats(feature)

        # Mean is 2.5, variance is ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4
        expected_variance = (2.25 + 0.25 + 0.25 + 2.25) / 4
        expected_std = math.sqrt(expected_variance)

        assert abs(stats.std_activation - expected_std) < 0.01

    def test_compute_token_entropy(self):
        fs = FeatureSet(layer=0)
        # All unique tokens - maximum entropy
        feature = self.create_feature_with_activations(
            0, ["a", "b", "c", "d"]
        )
        fs.add_feature(feature)

        analyzer = FeatureAnalyzer(fs)
        stats = analyzer.compute_activation_stats(feature)

        # Entropy of uniform distribution with 4 items = log2(4) = 2
        assert abs(stats.token_entropy - 2.0) < 0.01

    def test_compute_all_stats(self):
        fs = FeatureSet(layer=0)
        fs.add_feature(self.create_feature_with_activations(0, ["a", "b"]))
        fs.add_feature(self.create_feature_with_activations(1, ["c", "d"]))

        analyzer = FeatureAnalyzer(fs)
        all_stats = analyzer.compute_all_stats()

        assert 0 in all_stats
        assert 1 in all_stats

    def test_find_correlated_features(self):
        fs = FeatureSet(layer=0)

        # Two features that activate at same positions
        f1 = Feature(id=0, dimension=768)
        f2 = Feature(id=1, dimension=768)
        f3 = Feature(id=2, dimension=768)

        for i in range(5):
            act1 = FeatureActivation(
                feature_id=0, token="a", position=i,
                activation=1.0, context="ctx"
            )
            act2 = FeatureActivation(
                feature_id=1, token="b", position=i,
                activation=1.0, context="ctx"
            )
            f1.add_activation(act1)
            f2.add_activation(act2)

        # f3 activates at different positions
        for i in range(5, 10):
            act3 = FeatureActivation(
                feature_id=2, token="c", position=i,
                activation=1.0, context="ctx2"
            )
            f3.add_activation(act3)

        fs.add_feature(f1)
        fs.add_feature(f2)
        fs.add_feature(f3)

        analyzer = FeatureAnalyzer(fs)
        correlated = analyzer.find_correlated_features(0, min_correlation=0.5)

        # f2 should be correlated, f3 should not
        corr_ids = [fid for fid, _ in correlated]
        assert 1 in corr_ids
        assert 2 not in corr_ids

    def test_find_polysemantic_features(self):
        fs = FeatureSet(layer=0)

        # Feature with distinct token clusters
        feature = Feature(id=0, dimension=768)
        # Cluster 1: tokens starting with 'a'
        for i, token in enumerate(["apple", "ant", "and"]):
            act = FeatureActivation(
                feature_id=0, token=token, position=i, activation=1.0
            )
            feature.add_activation(act)
        # Cluster 2: tokens starting with 'b'
        for i, token in enumerate(["ball", "bat", "box"], start=3):
            act = FeatureActivation(
                feature_id=0, token=token, position=i, activation=1.0
            )
            feature.add_activation(act)

        fs.add_feature(feature)

        analyzer = FeatureAnalyzer(fs)
        polysemantic = analyzer.find_polysemantic_features(
            min_clusters=2, min_cluster_size=3
        )

        assert len(polysemantic) == 1
        assert polysemantic[0][0] == 0

    def test_get_dead_features(self):
        fs = FeatureSet(layer=0)
        fs.add_feature(Feature(id=0, dimension=768))  # Dead
        fs.add_feature(self.create_feature_with_activations(1, ["a"]))  # Active
        fs.add_feature(Feature(id=2, dimension=768))  # Dead

        analyzer = FeatureAnalyzer(fs)
        dead = analyzer.get_dead_features()

        assert 0 in dead
        assert 2 in dead
        assert 1 not in dead

    def test_get_ultra_sparse_features(self):
        fs = FeatureSet(layer=0)
        fs.total_tokens_processed = 1000000

        f1 = Feature(id=0, dimension=768)
        f1.activation_count = 10
        f1.activation_frequency = 10 / 1000000  # Ultra sparse
        fs.add_feature(f1)

        f2 = Feature(id=1, dimension=768)
        f2.activation_count = 10000
        f2.activation_frequency = 10000 / 1000000  # Not ultra sparse
        fs.add_feature(f2)

        analyzer = FeatureAnalyzer(fs)
        ultra_sparse = analyzer.get_ultra_sparse_features(max_frequency=0.0001)

        assert 0 in ultra_sparse
        assert 1 not in ultra_sparse

    def test_get_dense_features(self):
        fs = FeatureSet(layer=0)

        f1 = Feature(id=0, dimension=768)
        f1.activation_frequency = 0.2  # Dense
        fs.add_feature(f1)

        f2 = Feature(id=1, dimension=768)
        f2.activation_frequency = 0.05  # Not dense
        fs.add_feature(f2)

        analyzer = FeatureAnalyzer(fs)
        dense = analyzer.get_dense_features(min_frequency=0.1)

        assert 0 in dense
        assert 1 not in dense

    def test_compute_feature_importance(self):
        fs = FeatureSet(layer=0)

        f1 = Feature(id=0, dimension=768)
        f1.activation_frequency = 0.1
        f1.max_activation = 2.0
        fs.add_feature(f1)

        f2 = Feature(id=1, dimension=768)
        f2.activation_frequency = 0.01
        f2.max_activation = 1.0
        fs.add_feature(f2)

        analyzer = FeatureAnalyzer(fs)
        importance = analyzer.compute_feature_importance()

        # f1 should be more important
        assert importance[0][0] == 0  # Feature 0 first

    def test_get_position_distribution(self):
        fs = FeatureSet(layer=0)
        feature = Feature(id=0, dimension=768)

        for pos in [0, 1, 2, 10, 15, 30, 60]:
            act = FeatureActivation(
                feature_id=0, token="t", position=pos, activation=1.0
            )
            feature.add_activation(act)

        fs.add_feature(feature)

        analyzer = FeatureAnalyzer(fs)
        dist = analyzer.get_position_distribution(0)

        assert dist["start"] == 3  # positions 0, 1, 2
        assert dist["early"] == 2  # positions 10, 15
        assert dist["middle"] == 1  # position 30
        assert dist["late"] == 1  # position 60

    def test_compare_features(self):
        fs = FeatureSet(layer=0)
        f1 = self.create_feature_with_activations(0, ["hello", "world"])
        f2 = self.create_feature_with_activations(1, ["hello", "test"])
        f1.labels = ["greeting"]
        f2.labels = ["greeting", "test"]
        fs.add_feature(f1)
        fs.add_feature(f2)

        analyzer = FeatureAnalyzer(fs)
        comparison = analyzer.compare_features(0, 1)

        assert comparison["feature_1"] == 0
        assert comparison["feature_2"] == 1
        assert "hello" in comparison["token_overlap"]
        assert "greeting" in comparison["shared_labels"]
        assert comparison["jaccard_similarity"] > 0

    def test_get_summary(self):
        fs = FeatureSet(layer=0)
        fs.total_tokens_processed = 1000

        for i in range(10):
            f = Feature(id=i, dimension=768)
            f.activation_count = i * 10
            fs.add_feature(f)

        analyzer = FeatureAnalyzer(fs)
        summary = analyzer.get_summary()

        assert summary["total_features"] == 10
        assert "type_distribution" in summary
        assert "top_features" in summary
