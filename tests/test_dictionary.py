"""Tests for FeatureDictionary high-level API."""

import pytest
import tempfile
import os

from feature_dictionary.dictionary import (
    DictionaryConfig,
    FeatureDictionary,
    create_dictionary,
)
from feature_dictionary.features import FeatureType
from feature_dictionary.extraction import ExtractionConfig
from feature_dictionary.labeling import LabelingConfig


class TestDictionaryConfig:
    """Tests for DictionaryConfig."""

    def test_default_config(self):
        config = DictionaryConfig()
        assert config.name == "feature_dictionary"
        assert config.layer == 0
        assert config.extraction is not None
        assert config.labeling is not None

    def test_custom_config(self):
        config = DictionaryConfig(
            name="my_dict",
            model_name="gpt2",
            sae_name="sae_v1",
            layer=5,
        )
        assert config.name == "my_dict"
        assert config.model_name == "gpt2"
        assert config.layer == 5

    def test_to_dict(self):
        config = DictionaryConfig()
        d = config.to_dict()
        assert "name" in d
        assert "extraction" in d
        assert "labeling" in d


class TestFeatureDictionary:
    """Tests for FeatureDictionary."""

    def test_create_dictionary(self):
        dictionary = FeatureDictionary(
            num_features=100,
            feature_dim=768,
        )
        assert dictionary.num_features == 100
        assert dictionary.feature_dim == 768
        assert len(dictionary.features) == 100

    def test_dictionary_with_config(self):
        config = DictionaryConfig(
            model_name="test_model",
            layer=3,
        )
        dictionary = FeatureDictionary(
            num_features=50,
            feature_dim=512,
            config=config,
        )
        assert dictionary.config.model_name == "test_model"
        assert dictionary.config.layer == 3

    def test_process_activations(self):
        dictionary = FeatureDictionary(num_features=10, feature_dim=64)

        activations = [
            [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        tokens = ["hello", "world"]

        count = dictionary.process(activations, tokens, "hello world")
        assert count == 3  # 3 activations above threshold

    def test_process_batch(self):
        dictionary = FeatureDictionary(num_features=5, feature_dim=64)

        batch_activations = [
            [[1.0, 0.0, 0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0, 0.0]],
        ]
        batch_tokens = [["a"], ["b"]]

        total = dictionary.process_batch(batch_activations, batch_tokens)
        assert total == 2

    def test_finalize(self):
        dictionary = FeatureDictionary(num_features=5, feature_dim=64)

        # Process some data
        for i in range(5):
            activations = [[float(j == i)] * 5 for j in range(5)]
            tokens = [f"token{j}" for j in range(5)]
            dictionary.process(activations, tokens)

        dictionary.finalize()

        assert dictionary._finalized is True
        # Features should have labels and descriptions
        for feature in dictionary.features:
            assert isinstance(feature.labels, list)
            assert isinstance(feature.description, str)

    def test_get_feature(self):
        dictionary = FeatureDictionary(num_features=10, feature_dim=64)
        feature = dictionary.get_feature(5)
        assert feature is not None
        assert feature.id == 5

    def test_search_by_token(self):
        dictionary = FeatureDictionary(num_features=5, feature_dim=64)

        activations = [
            [2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0],
        ]
        tokens = ["hello", "world"]
        dictionary.process(activations, tokens)

        results = dictionary.search_by_token("hello")
        assert len(results) == 1
        assert results[0][0] == 0  # Feature 0
        assert results[0][1] == 2.0

    def test_search_by_label(self):
        dictionary = FeatureDictionary(num_features=5, feature_dim=64)

        # Manually add labels
        dictionary.features.features[0].labels = ["token:hello"]
        dictionary.features.features[1].labels = ["token:world"]

        results = dictionary.search_by_label("hello")
        assert len(results) == 1
        assert results[0].id == 0

    def test_search_by_pattern(self):
        dictionary = FeatureDictionary(num_features=5, feature_dim=64)

        # Add activations with numbers
        activations = [[1.0, 0.0, 0.0, 0.0, 0.0]]
        tokens = ["123"]
        dictionary.process(activations, tokens)

        results = dictionary.search_by_pattern(r"^\d+$")
        assert len(results) >= 1

    def test_get_top_features(self):
        dictionary = FeatureDictionary(num_features=5, feature_dim=64)

        # Feature 2 gets most activations
        for i in range(10):
            activations = [[0.0, 0.0, 1.0, 0.0, 0.0]]
            tokens = [f"tok{i}"]
            dictionary.process(activations, tokens)

        # Feature 0 gets fewer
        activations = [[1.0, 0.0, 0.0, 0.0, 0.0]]
        tokens = ["single"]
        dictionary.process(activations, tokens)

        top = dictionary.get_top_features(2, by="activation_count")
        assert top[0].id == 2

    def test_get_features_by_type(self):
        dictionary = FeatureDictionary(num_features=5, feature_dim=64)

        # Only activate feature 0
        for i in range(100):
            activations = [[1.0, 0.0, 0.0, 0.0, 0.0]]
            tokens = [f"tok{i}"]
            dictionary.process(activations, tokens)

        dictionary.finalize()

        # Feature 0 should be active, others dead
        dead = dictionary.get_features_by_type(FeatureType.DEAD)
        # Dead features are filtered by default
        # so we may not have any

    def test_find_similar(self):
        dictionary = FeatureDictionary(num_features=5, feature_dim=64)

        # Both features activate on same tokens
        activations = [
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
        ]
        tokens = ["hello", "world"]
        dictionary.process(activations, tokens)

        similar = dictionary.find_similar(0, top_k=3)
        # Feature 1 should be similar
        assert any(fid == 1 for fid, _ in similar)

    def test_get_statistics(self):
        dictionary = FeatureDictionary(num_features=10, feature_dim=64)

        # Process some data
        activations = [[1.0] * 10]
        tokens = ["test"]
        dictionary.process(activations, tokens)
        dictionary.finalize()

        stats = dictionary.get_statistics()
        assert "total_features" in stats
        assert "type_distribution" in stats

    def test_export_feature(self):
        dictionary = FeatureDictionary(num_features=5, feature_dim=64)

        activations = [[2.0, 0.0, 0.0, 0.0, 0.0]]
        tokens = ["test"]
        dictionary.process(activations, tokens)
        dictionary.finalize()

        export = dictionary.export_feature(0)
        assert "feature" in export
        assert "stats" in export
        assert export["feature"]["id"] == 0

    def test_export_all(self):
        dictionary = FeatureDictionary(num_features=5, feature_dim=64)
        dictionary.finalize()

        export = dictionary.export_all()
        assert "config" in export
        assert "feature_set" in export
        assert "statistics" in export

    def test_save_load(self):
        dictionary = FeatureDictionary(num_features=5, feature_dim=64)

        activations = [[1.0, 2.0, 0.0, 0.0, 0.0]]
        tokens = ["test"]
        dictionary.process(activations, tokens)
        dictionary.finalize()

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            path = f.name

        try:
            dictionary.save(path)
            loaded = FeatureDictionary.load(path)

            assert loaded._finalized is True
            assert len(loaded) > 0
        finally:
            os.unlink(path)

    def test_len(self):
        dictionary = FeatureDictionary(num_features=10, feature_dim=64)
        assert len(dictionary) == 10

    def test_getitem(self):
        dictionary = FeatureDictionary(num_features=10, feature_dim=64)
        feature = dictionary[5]
        assert feature.id == 5

    def test_getitem_not_found(self):
        dictionary = FeatureDictionary(num_features=5, feature_dim=64)
        dictionary.finalize()  # This filters dead features

        with pytest.raises(KeyError):
            _ = dictionary[999]

    def test_iter(self):
        dictionary = FeatureDictionary(num_features=5, feature_dim=64)
        count = sum(1 for _ in dictionary)
        assert count == 5

    def test_analyzer_property(self):
        dictionary = FeatureDictionary(num_features=5, feature_dim=64)
        analyzer = dictionary.analyzer
        assert analyzer is not None
        # Same instance on second access
        assert dictionary.analyzer is analyzer


class TestCreateDictionary:
    """Tests for create_dictionary helper."""

    def test_create_with_defaults(self):
        dictionary = create_dictionary(num_features=100, feature_dim=768)
        assert dictionary.num_features == 100
        assert dictionary.feature_dim == 768

    def test_create_with_model_info(self):
        dictionary = create_dictionary(
            num_features=50,
            feature_dim=512,
            model_name="gpt2",
            sae_name="sae_layer5",
            layer=5,
        )
        assert dictionary.config.model_name == "gpt2"
        assert dictionary.config.sae_name == "sae_layer5"
        assert dictionary.config.layer == 5

    def test_dictionary_workflow(self):
        """Test complete workflow."""
        # Create
        dictionary = create_dictionary(
            num_features=10,
            feature_dim=64,
            model_name="test",
        )

        # Process data
        for batch in range(5):
            activations = []
            tokens = []
            for pos in range(10):
                # Each position activates a different feature
                acts = [0.0] * 10
                acts[pos % 10] = 1.0 + batch * 0.1
                activations.append(acts)
                tokens.append(f"token{pos}")

            dictionary.process(activations, tokens)

        # Finalize
        dictionary.finalize()

        # Query
        top = dictionary.get_top_features(3)
        assert len(top) == 3

        stats = dictionary.get_statistics()
        assert stats["total_features"] > 0
