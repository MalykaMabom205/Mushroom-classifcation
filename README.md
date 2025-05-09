![image](https://github.com/user-attachments/assets/6940509d-6ad5-44b2-8397-f52e3f4f9fc7)


# Mushroom Classification Project

## Overview
This project applies machine learning techniques to classify mushrooms as either edible or poisonous using the Mushroom Classification dataset from Kaggle. The goal is to evaluate different classification algorithms and assess their performance using validation metrics.

## Summary of Work Done

### Data
The dataset was obtained from Kaggle's [Mushroom Classification dataset](https://www.kaggle.com/uciml/mushroom-classification). It contains:
- 8124 rows (mushroom instances)
- 22 categorical features
- 1 target column indicating edibility (`edible` or `poisonous`)

### Preprocessing / Cleanup
- Checked and handled missing values.
- Encoded categorical features using Label Encoding.
- Split the dataset into training (80%) and validation (20%) sets.

### Data Visualization
Visualizations were used to explore the dataset:
- Count plots of the edible vs. poisonous classes
- Histograms & bar plots of key features (odor, gill color, spore print color, etc.)

  ![image](https://github.com/user-attachments/assets/ede1c972-a057-498a-be5b-6594d8432994)
  ![image](https://github.com/user-attachments/assets/adaf02c0-5312-4052-a384-2060611132bd)
![image](https://github.com/user-attachments/assets/49691c67-48fa-44d9-a992-fcc52e52534f)
![image](https://github.com/user-attachments/assets/023dec0c-3669-4290-a81b-4301b2e687e4)
![image](https://github.com/user-attachments/assets/4bfea0a9-d030-423c-8b58-84731cb1dbde)
![image](https://github.com/user-attachments/assets/b31d0c27-257a-4006-bee5-b1a5dd61678d)
![image](https://github.com/user-attachments/assets/6b85f303-d671-4893-90d2-65c5077d7608)





## Problem Formulation
- Type: Binary Classification
- Input: Categorical features of mushrooms
- Output: `edible` or `poisonous`

## Confusion Matrix
![image](https://github.com/user-attachments/assets/15846741-393c-4c17-96b1-29a6090b4030)

Example confusion matrix from the best-performing model:

|                     | Predicted Edible | Predicted Poisonous |
|---------------------|------------------|----------------------|
| **Actual Edible**    | 843              | 0                    |
| **Actual Poisonous** | 0                | 782                  |


## Summary Metrics

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.998 |
| Precision  | 0.998 |
| Recall     | 0.998 |
| F1 Score   | 0.998 |
| ROC-AUC    | 0.999 |

## Training
The following machine learning models were trained:
- Decision Tree
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)

Each was evaluated using accuracy, precision, recall, F1 score, and ROC-AUC on the validation set.

## Performance Comparison

| Model                | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Decision Tree       | 0.996    | 0.996     | 0.996  | 0.996    | 0.997   |
| Random Forest       | 0.998    | 0.998     | 0.998  | 0.998    | 0.999   |
| Logistic Regression | 0.951    | 0.954     | 0.947  | 0.950    | 0.950   |
| SVM (RBF Kernel)    | 0.976    | 0.977     | 0.975  | 0.976    | 0.978   |

## Train ML Algorithm & Evaluate Performance on Validation Sample
Each model was trained on the training set and evaluated on the validation set. The Random Forest classifier achieved the best overall performance.

## Conclusion
The Random Forest model demonstrated superior performance in classifying mushrooms. Features like `odor` significantly contributed to accurate classification. Most models achieved high performance due to the dataset’s clarity and class separation.

## Future Work
- Apply ensemble models like Gradient Boosting or XGBoost
- Implement SHAP or permutation feature importance for interpretability
- Deploy the model through a web app or API
- Handle rare or ambiguous feature combinations more robustly

## Software Setup

### Requirements

Make sure you have the following installed:

- Python 3.8+
- Jupyter Notebook or JupyterLab
- pip or conda

### Installation

Clone the repository and install the required dependencies:

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/mushroom-classification.git
cd mushroom-classification
pip install -r requirements.txt    
```

## File Descriptions
| File / Folder          | Description                                                              |
| ---------------------- | ------------------------------------------------------------------------ |
| `MushroomKaggle.ipynb` | Jupyter Notebook containing all analysis, model training, and evaluation |
| `data/mushrooms.csv`   | Raw dataset downloaded from Kaggle                                       |
| `images/`              | Folder for generated plots and visualizations                            |
| `models/`              | Folder for saving trained models (optional)                              |
| `requirements.txt`     | List of required Python packages                                         |
| `README.md`            | Project documentation                                                    |


## Directory Structure
High-level structure:

MushroomKaggle.ipynb → Main Jupyter notebook

data/ → Contains the dataset file

images/ → Visualizations and plots

models/ → Trained model files (if saved)

requirements.txt → Python package dependencies

README.md → Project overview

## How to Reproduce Results
1. Download the dataset from Kaggle and place it in the data/ folder.

2. Install all dependencies using pip install -r requirements.txt.

3. Run the MushroomKaggle.ipynb notebook from start to finish.

4. Review metrics, confusion matrices, and performance comparisons.

## Citations
Kaggle Mushroom Classification dataset: https://www.kaggle.com/uciml/mushroom-classification
