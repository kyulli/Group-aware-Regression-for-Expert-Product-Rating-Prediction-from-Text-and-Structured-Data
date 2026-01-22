# Predicting Expert Coffee Ratings from Text and Structured Data  
### Group-Aware Regression and Interpretable Machine Learning

This repository presents an end-to-end applied machine learning project developed as the final project for DATA1030 (Hands-on Data Science) at Brown University. The project focuses on predicting expert coffee ratings by combining unstructured review text with structured product attributes, using group-aware validation and interpretable modeling techniques.

The accompanying PDF report documents the full methodology, empirical results, and model interpretation in detail.

## Project Overview

**Objective**  
To build a reproducible and interpretable regression pipeline that predicts expert coffee ratings on a 0 to 100 scale, while explicitly addressing non-independent data structure and model leakage.

**Why this matters**  
Expert product ratings are widely used in consumer decision making but are subjective, expensive, and difficult to scale. This project explores whether supervised learning models can approximate expert evaluations using both textual descriptions and structured product information, while maintaining statistical rigor.

**Key challenges addressed**
- Non-iid data due to repeated products and shared roasters
- High-dimensional sparse text features from TF-IDF
- Model interpretability for both structured and text-based predictors
- Reproducibility and leakage-safe evaluation

## Data

The dataset is derived from CoffeeReview and publicly available on Kaggle:  
https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset

**Dataset characteristics**
- 1,246 expert-reviewed specialty coffees (2017 to 2022)
- Review text written by professional tasters
- Structured attributes including origin, roast level, and price per 100g
- Target variable: expert rating score

The raw dataset is not stored in this repository due to size constraints. All preprocessing steps are fully reproducible using the provided notebooks.

## Methodology Highlights

**Feature Engineering**
- Review text processed with TF-IDF after tokenization, lemmatization, and stop-word removal
- Categorical variables encoded with one-hot encoding
- Numeric price feature standardized
- Missing categorical values explicitly preserved as informative signals

**Data Splitting**
- GroupShuffleSplit used for train test separation to prevent leakage
- GroupKFold used for cross-validation during hyperparameter tuning
- Entire pipeline repeated across multiple random states to assess robustness

**Models Evaluated**
- Ridge Regression
- Elastic Net
- Random Forest Regressor
- Support Vector Regression
- XGBoost Regressor with early stopping

**Evaluation**
- Root Mean Squared Error compared against a baseline predictor
- Improvement measured relative to baseline RMSE
- Best-performing model selected based on consistent out-of-sample performance

**Interpretability**
- Permutation feature importance
- Partial dependence analysis
- SHAP for global and local explanations

## Key Results

- Support Vector Regression achieved the strongest and most stable performance across test splits
- Review text was the dominant predictor, followed by roast level and price
- Price exhibited a clear positive marginal relationship with predicted rating
- Certain origins and textual descriptors consistently influenced predictions in interpretable ways

These findings and their implications are discussed in detail in the report.

## Repository Structure

project/

  ├── data/                # raw and preprocessed data (or link to dataset)
  
  ├── figures/             # generated plots
  
  ├── results/             # saved models, predictions, CV results
  
  ├── report/              # final PDF report
  
  ├── src/                 # all notebooks and python scripts
  
  ├── environment.yml      # reproducible environment
  
  ├── LICENSE
  
  └── README.md

## Dataset

The dataset originates from CoffeeReview and is available publicly on Kaggle: https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset

The dataset is not included directly in this repository due to size constraints. Instead, a download link is provided, and all preprocessing steps are fully reproducible through the `src/` notebooks.

## How to Reproduce the Results

### 1. Create and activate the environment

conda env create -f environment.yml
conda activate coffee_review_env

### 2. Run notebooks in `/src`
All preprocessing, TF-IDF vectorization, nested cross-validation, early stopping, and model evaluation steps are contained inside the notebooks and scripts.

### 3. Outputs
- Figures will be saved automatically to `/figures`
- Saved models, CV results, and predictions will appear in `/results`
- The final PDF report is located in `/report`

## Python and Package Versions

Key packages used:
- Python 3.10  
- numpy  
- pandas  
- scikit-learn  
- xgboost  
- matplotlib  
- seaborn  
- nltk

Full dependency list is provided in `environment.yml`.

## License
This project uses the MIT License.
