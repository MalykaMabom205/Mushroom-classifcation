![image](https://github.com/user-attachments/assets/55d7fc3d-c1ee-444b-9020-ecdbbddda81f)

# Mushroom Classification Project

## Overview

This project aims to classify mushrooms as **edible** or **poisonous** using machine learning techniques. The dataset contains 8,124 rows and 23 categorical features such as `cap shape`, `odor`, and `habitat`. The target column (`class`) indicates whether a mushroom is edible (`e`) or poisonous (`p`).


---

## Dataset

The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification). It contains the following key features:

| **Feature**       | **Description**                                                                 |
|--------------------|---------------------------------------------------------------------------------|
| `class`           | Edible (`e`) or Poisonous (`p`)                                                 |
| `cap-shape`       | Shape of the mushroom cap (e.g., bell, conical, convex)                         |
| `odor`            | Mushroom odor (e.g., almond, anise, foul)                                       |
| `population`      | Population size (e.g., abundant, clustered, scattered)                         |
| `habitat`         | Habitat of the mushroom (e.g., grasses, woods, urban)                          |

---

## Preprocessing

### Data Cleaning
- Missing values in the `stalk-root` column were imputed using **KNN Imputer**.
- The `veil-type` column was removed as it contained only one unique value.

### Encoding
- All categorical features were **one-hot encoded** to convert them into numerical format.
- The target column (`class`) was encoded as:
  - `0` for edible
  - `1` for poisonous

---

## Exploratory Data Analysis (EDA)

Key insights from the dataset:
- The `odor` feature is the most significant predictor of edibility. Mushrooms with a foul odor are almost always poisonous.
- Features like `cap color` and `habitat` also show noticeable patterns differentiating edible and poisonous mushrooms.

---

## Models Used

We implemented and evaluated the following machine learning models:

| **Model**                | **Validation Accuracy** | **Test Accuracy** | **Notes**                          |
|--------------------------|-------------------------|-------------------|-------------------------------------|
| Logistic Regression      | 99.75%                 | 99.75%            | Simple and interpretable.          |
| Random Forest Classifier | 100%                   | 100%              | Robust and provided feature importance. |
| XGBoost Classifier       | 100%                   | 100%              | High accuracy and performance.     |
| K-Nearest Neighbors      | 100%                   | 100%              | Computationally expensive.         |
| Support Vector Machines  | 100%                   | 100%              | High accuracy but less interpretable.|

The **Random Forest Classifier** was selected as the final model due to its perfect accuracy and interpretability.

---

## Results

The final model achieved **100% accuracy** on the test set, successfully distinguishing between edible and poisonous mushrooms.

---

## How to Reproduce

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Mushroom-Classification-Project.git
   cd Mushroom-Classification-Project
