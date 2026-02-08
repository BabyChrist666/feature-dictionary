"""Tests for core feature data structures."""

import pytest
import json
import tempfile
import os

from feature_dictionary.features import (
    FeatureType,
    FeatureActivation,
    Feature,
    FeatureSet,
)


class TestFeatureType:
    """Tests for FeatureType enum."""

    def test_types_exist(self):
        assert FeatureType.SPARSE
        assert FeatureType.MODERATE
        assert FeatureType.DENSE
        assert FeatureType.DEAD
        assert FeatureType.ULTRA_SPARSE

    def test_type_values(self):
        assert FeatureType.SPARSE.value == "sparse"
        assert FeatureType.DEAD.value == "dead"


class TestFeatureActivation:
    """Tests for FeatureActivation."""

    def test_create_activation(self):
        act = FeatureActivation(
            feature_id=42,
            token="hello",
            position=5,
            activation=1.5,
        )
        assert act.feature_id == 42
        assert act.token == "hello"
        assert act.position == 5
        assert act.activation == 1.5

    def test_activation_with_context(self):
        act = FeatureActivation(
            feature_id=1,
            token="world",
            position=10,
            activation=2.0,
            context="hello world example",
            context_tokens=["hello", "world", "example"],
        )
        assert act.context == "hello world example"
        assert len(act.context_tokens) == 3

    def test_to_dict(self):
        act = FeatureActivation(
            feature_id=1,
            token="test",
            position=0,
            activation=1.0,
        )
        d = act.to_dict()
        assert d["feature_id"] == 1
        assert d["token"] == "test"
        assert d["activation"] == 1.0

    def test_from_dict(self):
        data = {
            "feature_id": 5,
            "token": "example",
            "position": 3,
            "activation": 2.5,
            "context": "some context",
        }
        act = FeatureActivation.from_dict(data)
        assert act.feature_id == 5
        assert act.token == "example"
        assert act.activation == 2.5

    def test_roundtrip(self):
        act = FeatureActivation(
            feature_id=10,
            token="token",
            position=7,
            activation=3.0,
            context="context",
            context_tokens=["a", "b", "c"],
        )
        d = act.to_dict()
        restored = FeatureActivation.from_dict(d)
        assert restored.feature_id == act.feature_id
        assert restored.token == act.token
        assert restored.activation == act.activation


class TestFeature:
    """Tests for Feature."""

    def test_create_feature(self):
        feature = Feature(id=0, dimension=768)
        assert feature.id == 0
        assert feature.dimension == 768
        assert feature.activation_count == 0

    def test_add_activation(self):
        feature = Feature(id=0, dimension=768)
        act = FeatureActivation(
            feature_id=0,
            token="test",
            position=0,
            activation=1.5,
        )
        feature.add_activation(act)

        assert feature.activation_count == 1
        assert feature.mean_activation == 1.5
        assert feature.max_activation == 1.5
        assert len(feature.top_activations) == 1

    def test_add_multiple_activations(self):
        feature = Feature(id=0, dimension=768)

        for i in range(5):
            act = FeatureActivation(
                feature_id=0,
                token=f"token{i}",
                position=i,
                activation=float(i + 1),
            )
            feature.add_activation(act)

        assert feature.activation_count == 5
        assert feature.max_activation == 5.0
        # Mean of [1, 2, 3, 4, 5] = 3.0
        assert feature.mean_activation == 3.0

    def test_top_activations_limit(self):
        feature = Feature(id=0, dimension=768)

        for i in range(30):
            act = FeatureActivation(
                feature_id=0,
                token=f"token{i}",
                position=i,
                activation=float(i),
            )
            feature.add_activation(act, max_examples=10)

        # Should only keep top 10
        assert len(feature.top_activations) == 10
        # Highest activation should be 29.0
        assert feature.top_activations[0].activation == 29.0

    def test_classify_type_dead(self):
        feature = Feature(id=0, dimension=768)
        ftype = feature.classify_type(1000)
        assert ftype == FeatureType.DEAD

    def test_classify_type_sparse(self):
        feature = Feature(id=0, dimension=768)
        feature.activation_count = 5
        ftype = feature.classify_type(1000)
        assert ftype == FeatureType.SPARSE

    def test_classify_type_dense(self):
        feature = Feature(id=0, dimension=768)
        feature.activation_count = 200
        ftype = feature.classify_type(1000)
        assert ftype == FeatureType.DENSE

    def test_get_top_tokens(self):
        feature = Feature(id=0, dimension=768)
        for i, token in enumerate(["a", "b", "c"]):
            act = FeatureActivation(
                feature_id=0,
                token=token,
                position=i,
                activation=float(i + 1),
            )
            feature.add_activation(act)

        top = feature.get_top_tokens(2)
        assert len(top) == 2
        assert top[0][0] == "c"  # Highest activation

    def test_to_dict(self):
        feature = Feature(id=5, dimension=768, bias=0.1)
        feature.labels = ["test_label"]
        feature.description = "A test feature"

        d = feature.to_dict()
        assert d["id"] == 5
        assert d["dimension"] == 768
        assert d["labels"] == ["test_label"]

    def test_from_dict(self):
        data = {
            "id": 10,
            "dimension": 512,
            "bias": 0.5,
            "activation_count": 100,
            "labels": ["label1", "label2"],
            "feature_type": "sparse",
            "top_activations": [],
        }
        feature = Feature.from_dict(data)
        assert feature.id == 10
        assert feature.dimension == 512
        assert feature.activation_count == 100


class TestFeatureSet:
    """Tests for FeatureSet."""

    def test_create_feature_set(self):
        fs = FeatureSet(layer=5)
        assert fs.layer == 5
        assert len(fs) == 0

    def test_add_feature(self):
        fs = FeatureSet(layer=0)
        feature = Feature(id=0, dimension=768)
        fs.add_feature(feature)

        assert len(fs) == 1
        assert fs.get_feature(0) is feature

    def test_get_feature(self):
        fs = FeatureSet(layer=0)
        fs.add_feature(Feature(id=0, dimension=768))
        fs.add_feature(Feature(id=1, dimension=768))

        assert fs.get_feature(0) is not None
        assert fs.get_feature(1) is not None
        assert fs.get_feature(99) is None

    def test_getitem(self):
        fs = FeatureSet(layer=0)
        feature = Feature(id=42, dimension=768)
        fs.add_feature(feature)

        assert fs[42] is feature

    def test_iterate(self):
        fs = FeatureSet(layer=0)
        for i in range(5):
            fs.add_feature(Feature(id=i, dimension=768))

        count = sum(1 for _ in fs)
        assert count == 5

    def test_get_features_by_type(self):
        fs = FeatureSet(layer=0)
        for i in range(10):
            f = Feature(id=i, dimension=768)
            f.activation_count = i
            f.classify_type(100)
            fs.add_feature(f)

        dead = fs.get_features_by_type(FeatureType.DEAD)
        assert len(dead) == 1  # Only feature with 0 activations

    def test_get_features_by_label(self):
        fs = FeatureSet(layer=0)
        f1 = Feature(id=0, dimension=768)
        f1.labels = ["number", "digit"]
        f2 = Feature(id=1, dimension=768)
        f2.labels = ["letter"]
        fs.add_feature(f1)
        fs.add_feature(f2)

        number_features = fs.get_features_by_label("number")
        assert len(number_features) == 1
        assert number_features[0].id == 0

    def test_classify_all_types(self):
        fs = FeatureSet(layer=0)
        fs.total_tokens_processed = 1000

        for i in range(5):
            f = Feature(id=i, dimension=768)
            f.activation_count = 0  # Dead
            fs.add_feature(f)

        for i in range(5, 10):
            f = Feature(id=i, dimension=768)
            f.activation_count = 500  # Dense
            fs.add_feature(f)

        counts = fs.classify_all_types()
        assert counts[FeatureType.DEAD] == 5
        assert counts[FeatureType.DENSE] == 5

    def test_get_statistics(self):
        fs = FeatureSet(layer=0)
        fs.total_tokens_processed = 1000

        for i in range(10):
            f = Feature(id=i, dimension=768)
            f.activation_count = i * 10
            f.labels = ["label"] if i % 2 == 0 else []
            fs.add_feature(f)

        stats = fs.get_statistics()
        assert stats["total_features"] == 10
        assert stats["labeled_features"] == 5

    def test_save_load(self):
        fs = FeatureSet(layer=3, model_name="test_model")
        f = Feature(id=0, dimension=768)
        f.labels = ["test"]
        fs.add_feature(f)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
            path = tmp.name

        try:
            fs.save(path)
            loaded = FeatureSet.load(path)

            assert loaded.layer == 3
            assert loaded.model_name == "test_model"
            assert len(loaded) == 1
            assert loaded.get_feature(0).labels == ["test"]
        finally:
            os.unlink(path)

    def test_to_dict(self):
        fs = FeatureSet(layer=1, model_name="gpt2", sae_name="sae_v1")
        fs.add_feature(Feature(id=0, dimension=768))

        d = fs.to_dict()
        assert d["layer"] == 1
        assert d["model_name"] == "gpt2"
        assert d["sae_name"] == "sae_v1"
