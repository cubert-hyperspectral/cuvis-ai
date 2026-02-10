!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Nodes API

Complete API documentation for all node classes and implementations.

## Overview

Nodes are the building blocks of CUVIS.AI pipelines. This page documents all available node implementations organized by functional category.

---

## Anomaly Detection Nodes

Statistical and deep learning methods for detecting anomalies in hyperspectral data.

### RX Detector

::: cuvis_ai.anomaly.rx_detector
    options:
      show_root_heading: true
      heading_level: 4

### LAD Detector

::: cuvis_ai.anomaly.lad_detector
    options:
      show_root_heading: true
      heading_level: 4

### Deep SVDD

::: cuvis_ai.anomaly.deep_svdd
    options:
      show_root_heading: true
      heading_level: 4

---

## Binary Decision Nodes

Nodes that convert anomaly scores into binary decisions (anomaly/normal).

### Binary Decider

::: cuvis_ai.deciders.binary_decider
    options:
      show_root_heading: true
      heading_level: 4

### Two-Stage Decider

::: cuvis_ai.deciders.two_stage_decider
    options:
      show_root_heading: true
      heading_level: 4

---

## Data & Preprocessing Nodes

Nodes for data loading, normalization, and preprocessing.

### Data Loader

::: cuvis_ai.node.data
    options:
      show_root_heading: true
      heading_level: 4

### Normalization

::: cuvis_ai.node.normalization
    options:
      show_root_heading: true
      heading_level: 4

### Preprocessors

::: cuvis_ai.node.preprocessors
    options:
      show_root_heading: true
      heading_level: 4

### Conversion

::: cuvis_ai.node.conversion
    options:
      show_root_heading: true
      heading_level: 4

---

## Channel & Band Selection Nodes

Nodes for selecting and transforming spectral channels.

### Band Selection

::: cuvis_ai.node.band_selection
    options:
      show_root_heading: true
      heading_level: 4

### Channel Mixer

::: cuvis_ai.node.channel_mixer
    options:
      show_root_heading: true
      heading_level: 4

### Concrete Selector

::: cuvis_ai.node.concrete_selector
    options:
      show_root_heading: true
      heading_level: 4

### Channel Selector

::: cuvis_ai.node.selector
    options:
      show_root_heading: true
      heading_level: 4

---

## Deep Learning Nodes

Nodes implementing deep learning components.

### AdaCLIP

::: cuvis_ai.node.adaclip
    options:
      show_root_heading: true
      heading_level: 4

---

## Analysis & Dimensionality Reduction

Nodes for dimensionality reduction and feature extraction.

### PCA

::: cuvis_ai.node.pca
    options:
      show_root_heading: true
      heading_level: 4

---

## Visualization Nodes

Nodes for creating visualizations and TensorBoard logging.

### Visualizations

::: cuvis_ai.node.visualizations
    options:
      show_root_heading: true
      heading_level: 4

### TensorBoard Visualization

::: cuvis_ai.node.drcnn_tensorboard_viz
    options:
      show_root_heading: true
      heading_level: 4

### Monitor

::: cuvis_ai.node.monitor
    options:
      show_root_heading: true
      heading_level: 4

---

## Label Processing

Nodes for label conversion and manipulation.

### Labels

::: cuvis_ai.node.labels
    options:
      show_root_heading: true
      heading_level: 4

---

## Related Pages

- [Node System Deep Dive](../concepts/node-system-deep-dive.md)
- [Node Catalog](../node-catalog/index.md)
- [Add Built-in Node](../how-to/add-builtin-node.md)
