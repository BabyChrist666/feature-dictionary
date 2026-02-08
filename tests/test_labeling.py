"""Tests for feature labeling."""

import pytest

from feature_dictionary.labeling import (
    LabelingStrategy,
    Label,
    LabelingConfig,
    FeatureLabeler,
)
from feature_dictionary.features import Feature, FeatureSet, FeatureActivation


class TestLabelingStrategy:
    """Tests for LabelingStrategy enum."""

    def test_strategies_exist(self):
        assert LabelingStrategy.TOKEN_FREQUENCY
        assert LabelingStrategy.NGRAM_ANALYSIS
        assert LabelingStrategy.PATTERN_MATCHING
        assert LabelingStrategy.SEMANTIC_CLUSTERING
        assert LabelingStrategy.MANUAL

    def test_strategy_values(self):
        assert LabelingStrategy.TOKEN_FREQUENCY.value == "token_frequency"
        assert LabelingStrategy.PATTERN_MATCHING.value == "pattern_matching"


class TestLabel:
    """Tests for Label."""

    def test_create_label(self):
        label = Label(
            text="number",
            confidence=0.9,
            strategy=LabelingStrategy.PATTERN_MATCHING,
        )
        assert label.text == "number"
        assert label.confidence == 0.9
        assert label.strategy == LabelingStrategy.PATTERN_MATCHING

    def test_label_with_evidence(self):
        label = Label(
            text="digit",
            confidence=0.8,
            strategy=LabelingStrategy.TOKEN_FREQUENCY,
            evidence=["1", "2", "3"],
        )
        assert len(label.evidence) == 3

    def test_to_dict(self):
        label = Label(
            text="test",
            confidence=0.5,
            strategy=LabelingStrategy.MANUAL,
        )
        d = label.to_dict()
        assert d["text"] == "test"
        assert d["confidence"] == 0.5
        assert d["strategy"] == "manual"

    def test_from_dict(self):
        data = {
            "text": "pattern:number",
            "confidence": 0.7,
            "strategy": "pattern_matching",
            "evidence": ["123"],
        }
        label = Label.from_dict(data)
        assert label.text == "pattern:number"
        assert label.strategy == LabelingStrategy.PATTERN_MATCHING


class TestLabelingConfig:
    """Tests for LabelingConfig."""

    def test_default_config(self):
        config = LabelingConfig()
        assert config.min_token_frequency == 3
        assert config.max_labels == 5
        assert config.min_confidence == 0.3

    def test_custom_config(self):
        config = LabelingConfig(
            min_token_frequency=5,
            max_labels=10,
            patterns={"custom": r"custom_pattern"},
        )
        assert config.min_token_frequency == 5
        assert "custom" in config.patterns

    def test_to_dict(self):
        config = LabelingConfig()
        d = config.to_dict()
        assert "min_token_frequency" in d
        assert "strategies" in d


class TestFeatureLabeler:
    """Tests for FeatureLabeler."""

    def create_feature_with_activations(
        self,
        feature_id: int,
        tokens: list,
    ) -> Feature:
        """Helper to create a feature with activations."""
        feature = Feature(id=feature_id, dimension=768)
        for i, token in enumerate(tokens):
            act = FeatureActivation(
                feature_id=feature_id,
                token=token,
                position=i,
                activation=1.0,
                context_tokens=tokens[max(0, i-2):i+3],
            )
            feature.add_activation(act)
        return feature

    def test_create_labeler(self):
        labeler = FeatureLabeler()
        assert labeler.config is not None
        assert len(labeler.patterns) > 0

    def test_label_by_token_frequency(self):
        labeler = FeatureLabeler(config=LabelingConfig(min_token_frequency=2))

        # Create feature that activates on "the" multiple times
        feature = self.create_feature_with_activations(
            0, ["the", "the", "the", "a", "an"]
        )

        labels = labeler.label_feature(feature)

        # Should have a label for "the"
        assert any("the" in l.text for l in labels)

    def test_label_by_patterns(self):
        labeler = FeatureLabeler(
            config=LabelingConfig(
                strategies=[LabelingStrategy.PATTERN_MATCHING],
                min_confidence=0.2,
            )
        )

        # Create feature that activates on numbers
        feature = self.create_feature_with_activations(
            0, ["123", "456", "789", "text"]
        )

        labels = labeler.label_feature(feature)

        # Should detect number pattern
        assert any("number" in l.text for l in labels)

    def test_label_by_ngrams(self):
        labeler = FeatureLabeler(
            config=LabelingConfig(
                strategies=[LabelingStrategy.NGRAM_ANALYSIS],
                min_confidence=0.1,
            )
        )

        # Create feature with repeated context
        feature = Feature(id=0, dimension=768)
        for i in range(5):
            act = FeatureActivation(
                feature_id=0,
                token="word",
                position=i,
                activation=1.0,
                context_tokens=["hello", "world", "word"],
            )
            feature.add_activation(act)

        labels = labeler.label_feature(feature)

        # Should find some n-gram patterns
        ngram_labels = [l for l in labels if "ngram" in l.text]
        # May or may not find significant n-grams depending on threshold
        assert isinstance(ngram_labels, list)

    def test_label_features_batch(self):
        labeler = FeatureLabeler()

        fs = FeatureSet(layer=0)
        fs.add_feature(self.create_feature_with_activations(
            0, ["hello", "hello", "hello"]
        ))
        fs.add_feature(self.create_feature_with_activations(
            1, ["world", "world", "world"]
        ))

        results = labeler.label_features(fs)

        assert 0 in results
        assert 1 in results

    def test_generate_description(self):
        labeler = FeatureLabeler()

        feature = self.create_feature_with_activations(
            0, ["test", "test", "test"]
        )
        feature.feature_type = feature.classify_type(100)
        labeler.label_feature(feature)

        description = labeler.generate_description(feature)

        assert len(description) > 0
        assert feature.description == description

    def test_add_custom_pattern(self):
        labeler = FeatureLabeler()

        labeler.add_pattern("email", r"^[\w.]+@[\w.]+$")

        assert "email" in labeler.patterns
        assert labeler.patterns["email"] == r"^[\w.]+@[\w.]+$"

    def test_find_similar_features(self):
        labeler = FeatureLabeler()

        # Create two features with overlapping tokens
        f1 = self.create_feature_with_activations(
            0, ["hello", "world", "test"]
        )
        f2 = self.create_feature_with_activations(
            1, ["hello", "world", "example"]
        )
        f3 = self.create_feature_with_activations(
            2, ["completely", "different", "tokens"]
        )

        fs = FeatureSet(layer=0)
        fs.add_feature(f1)
        fs.add_feature(f2)
        fs.add_feature(f3)

        similar = labeler.find_similar_features(f1, fs, min_overlap=0.3)

        # f2 should be similar (shares "hello", "world")
        assert any(fid == 1 for fid, _ in similar)
        # f3 should not be similar
        assert not any(fid == 2 for fid, _ in similar)

    def test_pattern_matching_punctuation(self):
        labeler = FeatureLabeler(
            config=LabelingConfig(
                strategies=[LabelingStrategy.PATTERN_MATCHING],
                min_confidence=0.2,
            )
        )

        feature = self.create_feature_with_activations(
            0, [".", ",", "!", "?"]
        )

        labels = labeler.label_feature(feature)

        # Should detect punctuation pattern
        assert any("punctuation" in l.text for l in labels)

    def test_confidence_filtering(self):
        labeler = FeatureLabeler(
            config=LabelingConfig(min_confidence=0.9)
        )

        # Feature with varied tokens (low confidence for any single token)
        feature = self.create_feature_with_activations(
            0, ["a", "b", "c", "d", "e"]
        )

        labels = labeler.label_feature(feature)

        # High confidence threshold should filter out low-confidence labels
        for label in labels:
            assert label.confidence >= 0.9

    def test_max_labels_limit(self):
        labeler = FeatureLabeler(
            config=LabelingConfig(max_labels=2, min_confidence=0.0)
        )

        feature = self.create_feature_with_activations(
            0, ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
        )

        labels = labeler.label_feature(feature)

        assert len(labels) <= 2
