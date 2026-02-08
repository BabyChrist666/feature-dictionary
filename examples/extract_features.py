#!/usr/bin/env python3
"""
Feature Dictionary - Feature Extraction Example

This example demonstrates automated feature extraction and labeling
from Sparse Autoencoder (SAE) activations.
"""

import numpy as np
from feature_dictionary import (
    FeatureExtractor,
    FeatureLabeler,
    FeatureDictionary,
    ExtractionConfig,
    LabelingConfig,
)


def main():
    print("=" * 60)
    print("Feature Dictionary - Feature Extraction")
    print("=" * 60)

    # Create mock SAE activations
    n_samples = 1000
    n_features = 4096

    # Simulate sparse activations
    activations = np.random.exponential(1, (n_samples, n_features))
    activations *= (np.random.rand(n_samples, n_features) > 0.95)  # 95% sparse

    # Mock tokens for labeling
    tokens = [f"token_{i}" for i in range(n_samples)]

    # Example 1: Extract features
    print("\n1. Extracting features from activations...")

    config = ExtractionConfig(
        activation_threshold=0.1,
        min_activations=10,
        max_features=1000,
    )

    extractor = FeatureExtractor(config)
    features = extractor.extract(activations)

    print(f"   Extracted {len(features)} features")
    print(f"   Average activation frequency: {np.mean([f.frequency for f in features]):.4f}")

    # Example 2: Classify feature types
    print("\n2. Classifying feature types...")

    for feature in features[:5]:
        print(f"   Feature {feature.id}: type={feature.type}, "
              f"freq={feature.frequency:.4f}, max_act={feature.max_activation:.2f}")

    dead_features = [f for f in features if f.type == "dead"]
    sparse_features = [f for f in features if f.type == "sparse"]
    dense_features = [f for f in features if f.type == "dense"]

    print(f"\n   Dead features: {len(dead_features)}")
    print(f"   Sparse features: {len(sparse_features)}")
    print(f"   Dense features: {len(dense_features)}")

    # Example 3: Auto-label features
    print("\n3. Auto-labeling features...")

    labeling_config = LabelingConfig(
        use_top_tokens=True,
        top_k_tokens=10,
        use_ngrams=True,
    )

    labeler = FeatureLabeler(labeling_config)
    labeled_features = labeler.label(features, tokens, activations)

    for feature in labeled_features[:5]:
        print(f"   Feature {feature.id}: '{feature.label}'")
        print(f"      Top tokens: {feature.top_tokens[:3]}")

    # Example 4: Create feature dictionary
    print("\n4. Creating feature dictionary...")

    dictionary = FeatureDictionary()
    dictionary.add_features(labeled_features)

    print(f"   Dictionary size: {len(dictionary)}")
    print(f"   Unique labels: {len(dictionary.unique_labels)}")

    # Example 5: Search features
    print("\n5. Searching features by label...")

    # Search for features with specific patterns
    search_results = dictionary.search("token")
    print(f"   Found {len(search_results)} features matching 'token'")

    # Example 6: Get feature statistics
    print("\n6. Feature statistics...")

    stats = dictionary.get_statistics()
    print(f"   Mean frequency: {stats['mean_frequency']:.4f}")
    print(f"   Mean max activation: {stats['mean_max_activation']:.2f}")
    print(f"   Sparsity: {stats['sparsity']:.2%}")

    # Example 7: Export dictionary
    print("\n7. Exporting dictionary...")

    dictionary.save("feature_dictionary.json")
    print("   Saved to feature_dictionary.json")

    # Load it back
    loaded = FeatureDictionary.load("feature_dictionary.json")
    print(f"   Loaded dictionary with {len(loaded)} features")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
