# Feature Dictionary

[![Tests](https://github.com/BabyChrist666/feature-dictionary/actions/workflows/tests.yml/badge.svg)](https://github.com/BabyChrist666/feature-dictionary/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/BabyChrist666/feature-dictionary/branch/master/graph/badge.svg)](https://codecov.io/gh/BabyChrist666/feature-dictionary)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automated feature extraction and labeling from Sparse Autoencoders (SAEs). Extract meaningful features from neural network activations, label them with semantic descriptions, and analyze activation patterns.

## Features

- **Feature Extraction**: Process SAE activations and collect statistics
- **Automatic Labeling**: Multiple strategies for semantic labeling
- **Pattern Detection**: Identify token patterns, n-grams, and regex matches
- **Analysis Tools**: Correlations, polysemantic features, importance ranking
- **Serialization**: Save and load feature dictionaries

## Installation

```bash
pip install feature-dictionary
```

## Quick Start

```python
from feature_dictionary import FeatureDictionary, create_dictionary

# Create a dictionary for your SAE
dictionary = create_dictionary(
    num_features=4096,
    feature_dim=768,
    model_name="gpt2",
    sae_name="sae_layer5",
    layer=5,
)

# Process SAE activations
# activations: [seq_len, num_features]
# tokens: list of token strings
dictionary.process(activations, tokens, context="Original text")

# Finalize and compute labels
dictionary.finalize()

# Query features
feature = dictionary.get_feature(42)
print(f"Labels: {feature.labels}")
print(f"Description: {feature.description}")
print(f"Top tokens: {feature.get_top_tokens(5)}")

# Search by token
results = dictionary.search_by_token("hello")
for feature_id, activation in results:
    print(f"Feature {feature_id}: {activation}")
```

## Feature Extraction

### ExtractionConfig

```python
from feature_dictionary import FeatureExtractor, ExtractionConfig

config = ExtractionConfig(
    # Activation thresholds
    activation_threshold=0.0,
    min_activation=0.01,

    # Example collection
    max_examples_per_feature=20,
    context_window=5,

    # Filtering
    min_activation_count=1,
    include_dead_features=False,
)

extractor = FeatureExtractor(
    num_features=4096,
    feature_dim=768,
    config=config,
)

# Process activations
extractor.process_activations(activations, tokens)
extractor.process_batch(batch_activations, batch_tokens)

# Finalize
feature_set = extractor.finalize()
```

### Streaming Processing

```python
def activation_stream():
    for batch in data_loader:
        sae_activations = sae.encode(batch.hidden_states)
        yield sae_activations, batch.tokens

for count in extractor.stream_activations(activation_stream()):
    print(f"Recorded {count} activations")
```

## Feature Labeling

### LabelingConfig

```python
from feature_dictionary import FeatureLabeler, LabelingConfig, LabelingStrategy

config = LabelingConfig(
    # Strategies to use
    strategies=[
        LabelingStrategy.TOKEN_FREQUENCY,
        LabelingStrategy.NGRAM_ANALYSIS,
        LabelingStrategy.PATTERN_MATCHING,
    ],

    # Token frequency settings
    min_token_frequency=3,
    max_labels=5,

    # Confidence threshold
    min_confidence=0.3,

    # Custom patterns
    patterns={
        "email": r"^[\w.]+@[\w.]+$",
        "url": r"^https?://",
    },
)

labeler = FeatureLabeler(config)

# Label a feature
labels = labeler.label_feature(feature)
for label in labels:
    print(f"{label.text}: {label.confidence:.2f}")

# Generate description
description = labeler.generate_description(feature)
print(description)
```

### Labeling Strategies

- **TOKEN_FREQUENCY**: Most common tokens
- **NGRAM_ANALYSIS**: Significant n-gram patterns
- **PATTERN_MATCHING**: Regex pattern matching
- **SEMANTIC_CLUSTERING**: Embedding-based clustering
- **MANUAL**: Human-provided labels

### Built-in Patterns

```python
# Pre-defined patterns
patterns = {
    "number": r"^\d+$",
    "punctuation": r"^[^\w\s]+$",
    "uppercase": r"^[A-Z]+$",
    "code_keyword": r"^(if|else|for|while|def|class)$",
    "whitespace": r"^\s+$",
    "article": r"^(a|an|the)$",
    "pronoun": r"^(i|you|he|she|it|we|they)$",
    "preposition": r"^(in|on|at|to|for|with)$",
}
```

## Feature Analysis

```python
from feature_dictionary import FeatureAnalyzer

analyzer = FeatureAnalyzer(feature_set)

# Compute statistics
stats = analyzer.compute_activation_stats(feature)
print(f"Mean: {stats.mean_activation}")
print(f"Std: {stats.std_activation}")
print(f"Entropy: {stats.token_entropy}")

# Find correlations
correlated = analyzer.find_correlated_features(feature_id=42)
for fid, correlation in correlated:
    print(f"Feature {fid}: {correlation:.2f}")

# Find polysemantic features
polysemantic = analyzer.find_polysemantic_features(
    min_clusters=2,
    min_cluster_size=3,
)

# Get feature types
dead = analyzer.get_dead_features()
sparse = analyzer.get_ultra_sparse_features()
dense = analyzer.get_dense_features()

# Importance ranking
importance = analyzer.compute_feature_importance()
for fid, score in importance[:10]:
    print(f"Feature {fid}: {score:.3f}")
```

## Feature Types

```python
from feature_dictionary import FeatureType

# Classification based on activation frequency
FeatureType.DEAD          # Never activates
FeatureType.ULTRA_SPARSE  # < 0.01% frequency
FeatureType.SPARSE        # < 1% frequency
FeatureType.MODERATE      # 1-10% frequency
FeatureType.DENSE         # > 10% frequency
```

## Dictionary API

```python
# Search
dictionary.search_by_token("hello")
dictionary.search_by_label("number")
dictionary.search_by_pattern(r"^\d+$")

# Get top features
top = dictionary.get_top_features(10, by="activation_count")
top = dictionary.get_top_features(10, by="max_activation")
top = dictionary.get_top_features(10, by="frequency")

# Find similar features
similar = dictionary.find_similar(feature_id=42, top_k=5)

# Get correlated features
correlated = dictionary.get_correlated(feature_id=42)

# Statistics
stats = dictionary.get_statistics()
print(f"Total: {stats['total_features']}")
print(f"Active: {stats['active_features']}")
print(f"Labeled: {stats['labeled_features']}")
```

## Serialization

```python
# Save dictionary
dictionary.save("features.json")

# Load dictionary
dictionary = FeatureDictionary.load("features.json")

# Export single feature
export = dictionary.export_feature(42)

# Export all
export = dictionary.export_all()
```

## Integration with SAE

```python
from feature_dictionary import FeatureDictionary

# With TransformerLens SAE
sae = load_sae("sae_model.pt")
dictionary = FeatureDictionary(
    num_features=sae.num_features,
    feature_dim=sae.d_in,
)

# Process model outputs
for batch in dataloader:
    hidden_states = model(batch.input_ids)
    sae_activations = sae.encode(hidden_states)
    dictionary.process(
        sae_activations.tolist(),
        tokenizer.convert_ids_to_tokens(batch.input_ids),
    )

dictionary.finalize()
```

## API Reference

### Core Classes

- `Feature` - Single SAE feature with activation data
- `FeatureSet` - Collection of features from a layer
- `FeatureActivation` - Single activation record
- `FeatureType` - Feature classification enum

### Extraction

- `FeatureExtractor` - Extracts features from activations
- `ExtractionConfig` - Extraction configuration
- `MockSAE` - Mock SAE for testing

### Labeling

- `FeatureLabeler` - Assigns semantic labels
- `LabelingConfig` - Labeling configuration
- `Label` - Semantic label with confidence
- `LabelingStrategy` - Labeling strategy enum

### Analysis

- `FeatureAnalyzer` - Analyzes feature patterns
- `ActivationStats` - Activation statistics

### Dictionary

- `FeatureDictionary` - High-level API
- `DictionaryConfig` - Dictionary configuration
- `create_dictionary()` - Helper function

## License

MIT

