# Naturalis Multimodal
Multimodal Learning for Biodiversity Taxonomic Metadata Completion


The code in this repositroy is for the MSc "Multimodal Models for Biodiversity Taxonomic Metadata Completion".

This project explores multimodal learning models for the automation of taxonomic metadata completion. The dataset used for this project is the Cainozoic Mollusca dataset from Naturalis Biodiversity Center, publicly accessible through GBIF. The task of metadata completion is framed as molluscan order multi-class classification, in order to compare unimodal vs. multimodal approaches. Image-only BioCLIP2, metadata-only ModernBERT and multimodal weighted and gated fusion models are compared under progressively restricted metadata conditions.

---

## Repository Structure

### `majoritybaseline`
Scripts for the majority-class baseline used for comparison.

### `imageonlymodel`
Scripts for image processing, BioCLIP2 embedding extraction, and image-only classification.

### `metadataonlymodel`
Scripts for ModernBERT metadata classifier and taxonomic prediction from specimen metadata.

### `multimodalfusionmodels`
Scripts for the multimodal weighted and gated fusion models combining image and metadata representations.

### `evaluationmetrics`
Scripts used to compute evaluation metrics, rarity analyses, significance testing, and bootstrap confidence intervals.

### `exploratoryphi4mm`
Scripts for exploratory experiments using Phi-4 Multimodal-Instruct for selected ultra-rare specimens and challenging prediction cases.

### `EDAFINAL.ipynb`
Script for the Exploratory Data Analysis (EDA) notebook documenting dataset exploration, missing data analysis, taxonomic leakage analysis, and dataset preparation.

---

## Dataset

THe Naturalis Biodiversity Center Cainozoic Mollusca dataset contains digitized molluscan fossil specimen records accessible through GBIF.

---

## Models

The repository contains implementations of:

- Majority baseline
- BioCLIP2 image-only classifier
- ModernBERT metadata-only classifier
- Multimodal weighted and gated fusion
- Exploratory Phi-4 Multimodal analysis

---

## Evaluation

Models were evaluated using:

- Top-1 Accuracy
- Top-3 Accuracy
- Balanced Accuracy
- Macro-F1
- McNemar significance tests (Holm's corrected)
- Bootstrap confidence intervals
- Cohen's κ for cross-modal agreement

---

## Main Result

The strongest model was the multimodal gated fusion model.

Under the realistic **Order-Cleaned Metadata** condition it achieved:

| Metric | Score |
|---------|------:|
| Top-1 Accuracy | 92.2% |
| Top-3 Accuracy | 98.4% |
| Balanced Accuracy | 76.0% |
| Macro-F1 | 0.763 |

---


## Environment

Experiments were implemented using Python together with:

- PyTorch
- Transformers
- OpenCLIP
- scikit-learn
- pandas
- NumPy

Training was performed on the Dutch national supercomputer **Snellius** using NVIDIA A100 GPUs.

---
## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

The main model experiments were developed and executed on the Dutch national supercomputer Snellius using Python 3.10.

## Note

Overall, this repository accompanies the experimental workflow used to obtain the reported results and evaluation metrics for taxonomic metadata completion.
