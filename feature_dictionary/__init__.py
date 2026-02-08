"""
Feature Dictionary: Automated Feature Extraction and Labeling from SAE

A toolkit for extracting, analyzing, and labeling features from Sparse Autoencoders.
"""

from .features import Feature, FeatureSet, FeatureActivation
from .extraction import FeatureExtractor, ExtractionConfig
from .labeling import FeatureLabeler, LabelingConfig, Label
from .analysis import FeatureAnalyzer, ActivationStats
from .dictionary import FeatureDictionary

__version__ = "0.1.0"

__all__ = [
    "Feature",
    "FeatureSet",
    "FeatureActivation",
    "FeatureExtractor",
    "ExtractionConfig",
    "FeatureLabeler",
    "LabelingConfig",
    "Label",
    "FeatureAnalyzer",
    "ActivationStats",
    "FeatureDictionary",
]
