# Predicting Coffee Review Ratings Using TF-IDF and Regularized Regression Models

This repository contains the full implementation, data processing pipeline, modeling code, and final report for my DATA1030 final project at Brown University. The objective is to predict expert coffee ratings using a combination of cleaned review text (TF-IDF) and structured attributes such as origin, roast level, price, and variety.   The project emphasizes reproducibility with a consistent environment, organized directory structure, and version-controlled experiments.

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
conda activate coffee-rating-env

### 2. Run notebooks in `/src`
All preprocessing, TF-IDF vectorization, nested cross-validation, early stopping, and model evaluation steps are contained inside the notebooks and scripts.

### 3. Outputs
- Figures will be saved automatically to `/figures`
- Saved models, CV results, and predictions will appear in `/results`
- The final PDF report is located in `/report`

## Python and Package Versions

Key packages used:
- Python 3.13.5  
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
