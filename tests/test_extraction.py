"""Tests for feature extraction."""

import pytest

from feature_dictionary.extraction import (
    ExtractionConfig,
    FeatureExtractor,
    MockSAE,
)
from feature_dictionary.features import FeatureType


class TestExtractionConfig:
    """Tests for ExtractionConfig."""

    def test_default_config(self):
        config = ExtractionConfig()
        assert config.activation_threshold == 0.0
        assert config.min_activation == 0.01
        assert config.max_examples_per_feature == 20
        assert config.context_window == 5

    def test_custom_config(self):
        config = ExtractionConfig(
            min_activation=0.1,
            max_examples_per_feature=50,
            include_dead_features=True,
        )
        assert config.min_activation == 0.1
        assert config.max_examples_per_feature == 50
        assert config.include_dead_features is True

    def test_to_dict(self):
        config = ExtractionConfig()
        d = config.to_dict()
        assert "activation_threshold" in d
        assert "min_activation" in d
        assert "batch_size" in d


class TestFeatureExtractor:
    """Tests for FeatureExtractor."""

    def test_create_extractor(self):
        extractor = FeatureExtractor(
            num_features=100,
            feature_dim=768,
            layer=5,
        )
        assert extractor.num_features == 100
        assert extractor.feature_dim == 768
        assert extractor.layer == 5
        assert len(extractor.features) == 100

    def test_process_activations(self):
        extractor = FeatureExtractor(
            num_features=10,
            feature_dim=64,
        )

        # Create mock activations
        activations = [
            [0.5, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        tokens = ["hello", "world", "test"]

        num_recorded = extractor.process_activations(activations, tokens)

        assert num_recorded == 4  # 4 activations above threshold
        assert extractor._total_tokens == 3

    def test_process_with_context(self):
        extractor = FeatureExtractor(
            num_features=5,
            feature_dim=64,
        )

        activations = [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]
        tokens = ["a", "b", "c"]

        extractor.process_activations(
            activations, tokens,
            context="a b c"
        )

        # Check feature 0 has activation with context
        feature = extractor.features.get_feature(0)
        assert len(feature.top_activations) == 1
        assert feature.top_activations[0].context == "a b c"

    def test_process_batch(self):
        extractor = FeatureExtractor(
            num_features=5,
            feature_dim=64,
        )

        batch_activations = [
            [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]],
        ]
        batch_tokens = [
            ["token1", "token2"],
            ["token3", "token4"],
        ]

        total = extractor.process_batch(batch_activations, batch_tokens)
        assert total == 4
        assert extractor._total_tokens == 4

    def test_finalize(self):
        extractor = FeatureExtractor(
            num_features=10,
            feature_dim=64,
            config=ExtractionConfig(include_dead_features=False),
        )

        # Process some activations (only feature 0 activates)
        activations = [[1.0] + [0.0] * 9] * 5
        tokens = ["token"] * 5

        extractor.process_activations(activations, tokens)
        feature_set = extractor.finalize()

        # Only feature 0 should remain (dead features filtered)
        assert len(feature_set) == 1
        assert 0 in feature_set.features

    def test_finalize_keep_dead(self):
        extractor = FeatureExtractor(
            num_features=10,
            feature_dim=64,
            config=ExtractionConfig(include_dead_features=True, min_activation_count=0),
        )

        activations = [[1.0] + [0.0] * 9] * 5
        tokens = ["token"] * 5

        extractor.process_activations(activations, tokens)
        feature_set = extractor.finalize()

        # All features should remain
        assert len(feature_set) == 10

    def test_get_top_features(self):
        extractor = FeatureExtractor(
            num_features=5,
            feature_dim=64,
        )

        # Feature 2 activates most (10 times)
        for i in range(10):
            activations = [[0.0, 0.0, 1.0, 0.0, 0.0]]
            tokens = [f"token{i}"]
            extractor.process_activations(activations, tokens)

        # Feature 0 activates less (3 times)
        for i in range(3):
            activations = [[1.0, 0.0, 0.0, 0.0, 0.0]]
            tokens = [f"other{i}"]
            extractor.process_activations(activations, tokens)

        top = extractor.get_top_features(1)
        assert len(top) == 1
        assert top[0].id == 2

    def test_get_feature_by_token(self):
        extractor = FeatureExtractor(
            num_features=5,
            feature_dim=64,
        )

        activations = [
            [1.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0],
        ]
        tokens = ["hello", "world"]

        extractor.process_activations(activations, tokens)

        results = extractor.get_feature_by_token("hello")
        assert len(results) == 1
        assert results[0][0] == 0  # Feature 0
        assert results[0][1] == 1.5

    def test_callback(self):
        extractor = FeatureExtractor(
            num_features=3,
            feature_dim=64,
        )

        recorded = []
        extractor.add_callback(lambda act: recorded.append(act))

        activations = [[1.0, 2.0, 0.0]]
        tokens = ["test"]
        extractor.process_activations(activations, tokens)

        assert len(recorded) == 2

    def test_stream_activations(self):
        extractor = FeatureExtractor(
            num_features=3,
            feature_dim=64,
        )

        def activation_stream():
            for i in range(5):
                yield [[1.0, 0.0, 0.0]], [f"token{i}"]

        counts = list(extractor.stream_activations(activation_stream()))
        assert len(counts) == 5
        assert all(c == 1 for c in counts)

    def test_context_window(self):
        extractor = FeatureExtractor(
            num_features=3,
            feature_dim=64,
            config=ExtractionConfig(context_window=2),
        )

        activations = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Feature 0 activates at position 2
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        tokens = ["a", "b", "c", "d", "e"]

        extractor.process_activations(activations, tokens)

        feature = extractor.features.get_feature(0)
        act = feature.top_activations[0]
        # Context window of 2 around position 2: [a, b, c, d, e][0:5]
        assert "c" in act.context_tokens


class TestMockSAE:
    """Tests for MockSAE."""

    def test_create_mock_sae(self):
        sae = MockSAE(input_dim=768, num_features=100, sparsity=0.1)
        assert sae.input_dim == 768
        assert sae.num_features == 100
        assert sae.sparsity == 0.1

    def test_encode(self):
        sae = MockSAE(input_dim=64, num_features=32, sparsity=0.2)

        # Mock input
        x = [[0.5] * 64 for _ in range(10)]

        output = sae.encode(x)
        assert len(output) == 10
        assert len(output[0]) == 32

        # Check sparsity (roughly 20% active)
        active_count = sum(1 for a in output[0] if a > 0)
        assert active_count > 0
        assert active_count < 32

    def test_encode_produces_sparse_activations(self):
        sae = MockSAE(input_dim=64, num_features=100, sparsity=0.05)

        x = [[0.5] * 64 for _ in range(5)]
        output = sae.encode(x)

        # Most features should be zero
        for acts in output:
            zero_count = sum(1 for a in acts if a == 0.0)
            assert zero_count > 50  # More than half should be zero
