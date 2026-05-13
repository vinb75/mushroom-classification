# Mushroom Species Classification from Field Photographs

**CBS Copenhagen Business School — Machine Learning and Deep Learning, Spring 2026**

## Research Question

*How does model complexity affect the classification of mushroom species from field photographs?*

- **Sub-RQ1:** At what level of complexity do classifiers achieve reliable species identification?
- **Sub-RQ2:** How do misclassification patterns differ across model types in terms of toxicity risk?

## Dataset

- **Source:** iNaturalist Denmark
- **Size:** ~9,200 field photographs across 19 mushroom species
- **Split:** Observer-aware train/val/test split (zero photographer overlap between partitions)
- **Images:** Hosted on [GitHub Releases](https://github.com/vinb75/mushroom-classification/releases/tag/v1.0) and downloaded automatically by the notebook

## Models

| Role | Model | Test Accuracy |
|------|-------|---------------|
| Main Model 1 | Logistic Regression (PCA features) | 30.6% |
| Main Model 2 | Random Forest (PCA features) | 29.2% |
| Main Model 3 | Custom CNN (4 variants, trained from scratch) | 59.1% |
| Benchmark Baseline | EfficientNet-B0 (pretrained, fine-tuned) | 92.3% |

## Key Findings

1. **Model complexity does not monotonically improve performance.** Random Forest (582K parameters) performs worse than Logistic Regression (1.9K parameters). The critical factor is feature quality, not parameter count.
2. **Custom CNN and pretrained B0 have nearly identical overfitting rates** (~7.5pp train-test gap), despite a 33pp accuracy difference. The gap is driven by feature quality from ImageNet pretraining, not generalisation ability.
3. **Regularisation is essential for from-scratch CNN training.** Without BatchNorm and Dropout, the custom CNN fails to learn entirely (6.6% accuracy ≈ random chance).
4. **The confidence paradox:** EfficientNet-B0 produces the fewest dangerous misclassifications but its dangerous errors are the most confidently wrong (80.7% mean confidence), meaning confidence thresholds alone cannot ensure safety.

## Repository Structure

```
mushroom-classification/
├── ML_project.ipynb          # Single notebook (all code)
├── README.md
├── requirements.txt
├── data/
│   ├── splits/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   ├── metadata_enriched.csv
│   ├── species_list.csv
│   └── toxicity_table.csv
├── models/                   # Saved model files (after training)
└── results/                  # Output figures and prediction CSVs
```

## How to Run

### Option 1: Google Colab (recommended)
1. Open `ML_project.ipynb` in Google Colab
2. Click **Run All**
3. The notebook automatically clones the repo and downloads images

### Option 2: Local
```bash
git clone https://github.com/vinb75/mushroom-classification.git
cd mushroom-classification
pip install -r requirements.txt
jupyter notebook ML_project.ipynb
# Run all cells — images download automatically on first run
```

## Authors

Vincent Ballwieser, Peter Szalay, Connor Kennedy

## License

This project is for academic purposes (CBS Spring 2026).