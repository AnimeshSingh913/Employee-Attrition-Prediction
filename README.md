# Employee Attrition Prediction: Decision Tree vs Random Forest

A machine learning project built with Python, scikit-learn, and pandas to predict employee attrition using the IBM HR Analytics dataset, comparing performance between a Decision Tree Classifier and a Random Forest Classifier (100 estimators).

---

## Objective
The goal of this project is to build a complete, end-to-end reproducible ML pipeline to predict whether an employee will leave an organization (`Attrition`: Yes/No). We evaluate and contrast a single unconstrained **Decision Tree Classifier** against a **Random Forest Classifier** (100 estimators) on key classification metrics, feature importances, and hyperparameter tuning effects.

---

## Dataset Link
The dataset used in this project is the **IBM HR Analytics Employee Attrition & Performance Dataset**.
- **Kaggle URL:** [IBM HR Analytics Attrition Dataset on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Dataset Details:** Contains 1,470 employee records and 35 columns covering demographic details, job role, monthly income, overtime status, and work-life balance.

*Note: As per repository rules, raw dataset CSV files are gitignored and not included in this repository.*

---

## Libraries Used
- **Python 3.10+**
- **pandas**: Data manipulation, inspection, and one-hot encoding
- **numpy**: Numerical operations
- **scikit-learn**: Data splitting (`train_test_split`), `DecisionTreeClassifier`, `RandomForestClassifier`, and evaluation metrics (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`)
- **matplotlib**: Visualization layout and plot exporting
- **seaborn**: Confusion matrix heatmaps and feature importance bar charts
- **jupyter**: Notebook execution environment

---

## Methodology

1. **Data Understanding**:
   - Loaded raw dataset (1,470 rows × 35 columns).
   - Programmatically derived 26 numerical features, 8 categorical features, and 1 target variable (`Attrition`).
   - Analyzed class imbalance: ~83.88% No Attrition (1,233 rows) vs ~16.12% Yes Attrition (237 rows).

2. **Data Preprocessing**:
   - Confirmed 0 missing values across all features.
   - Programmatically detected and dropped 3 constant columns (`EmployeeCount`, `Over18`, `StandardHours` where `nunique() == 1`) and 1 uninformative ID column (`EmployeeNumber`).
   - Binary encoded target variable (`Attrition`: `Yes` → 1, `No` → 0).
   - Applied one-hot encoding (`pd.get_dummies(..., drop_first=True)`) to categorical nominal features, resulting in 44 predictor columns.
   - Performed 80/20 train/test split with stratification (`random_state=42`), producing 1,176 training samples and 294 test samples.

3. **Model Development**:
   - **Model 1**: `DecisionTreeClassifier(random_state=42)`
   - **Model 2**: `RandomForestClassifier(n_estimators=100, random_state=42)`
   - Fitted both models on identical training data and predicted on identical test data.

4. **Model Evaluation & Comparison**:
   - Evaluated models using Accuracy, Precision, Recall, and F1 Score (focusing on positive label `1` for Attrition).
   - Generated labeled side-by-side confusion matrix heatmaps saved to `outputs/confusion_matrices.png`.
   - Extracted and plotted top 15 Random Forest feature importances saved to `outputs/feature_importance.png`.

5. **Hyperparameter Tuning (Bonus)**:
   - Evaluated Decision Tree across different maximum tree depths (`max_depth` = 3, 5, 10, None) to quantify depth pruning impact on variance and out-of-sample accuracy.

---

## Results

### Performance Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree** | 0.7653 | 0.3103 | 0.3830 | 0.3429 |
| **Random Forest (100 Trees)** | **0.8333** | **0.4167** | 0.1064 | 0.1695 |

### Confusion Matrices
Saved to `outputs/confusion_matrices.png`:
- **Decision Tree**: True Negatives = 207, False Positives = 40, False Negatives = 29, True Positives = 18
- **Random Forest**: True Negatives = 240, False Positives = 7, False Negatives = 42, True Positives = 5

### Top 15 Feature Importances (Random Forest)
Saved to `outputs/feature_importance.png`:
1. `MonthlyIncome`
2. `OverTime_Yes`
3. `Age`
4. `TotalWorkingYears`
5. `DailyRate`
6. `MonthlyRate`
7. `HourlyRate`
8. `DistanceFromHome`
9. `YearsAtCompany`
10. `NumCompaniesWorked`
11. `PercentSalaryHike`
12. `StockOptionLevel`
13. `YearsWithCurrManager`
14. `YearsInCurrentRole`
15. `JobSatisfaction`

### Hyperparameter Experiment (Decision Tree max_depth)

| Tree Configuration | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **max_depth = 3** | **0.8333** | **0.4545** | 0.2128 | 0.2899 |
| **max_depth = 5** | **0.8333** | 0.4444 | 0.1702 | 0.2462 |
| **max_depth = 10** | 0.8027 | 0.3778 | 0.3617 | **0.3696** |
| **max_depth = None (Default)** | 0.7653 | 0.3103 | **0.3830** | 0.3429 |

---

## Model Comparison

1. **Accuracy & Precision Advantage**: Random Forest achieves higher overall classification accuracy (**83.33%** vs **76.53%**) and precision (**41.67%** vs **31.03%**). It produces far fewer false positive attrition alarms (7 vs 40).
2. **Recall Trade-off in Class Imbalance**: Due to dataset class imbalance (~16% positive class) and standard 0.5 decision thresholding, Random Forest exhibits lower recall (**10.64%** vs **38.30%**).
3. **Key Predictive Drivers**: Both models highlight financial factors (`MonthlyIncome`), workload indicators (`OverTime`), and tenure/experience (`Age`, `TotalWorkingYears`) as primary drivers of employee attrition.

---

## Conclusion

Random Forest (100 trees) outperformed the default Decision Tree on accuracy by **+6.80%** (83.33% vs 76.53%) and precision by **+10.64%** (41.67% vs 31.03%). 

### Why Random Forest Outperforms Decision Tree
Random Forest reduces variance through **bagging (bootstrap aggregation)** and feature randomization. By averaging predictions across 100 decorrelated trees, it smooths out noisy decision boundaries inherent in single trees.

### Core Model Limitations
- **Decision Tree Limitation**: Highly susceptible to **overfitting**. Without depth constraints, single decision trees fit training noise, leading to higher test error (76.53% accuracy).
- **Random Forest Limitation**: Operates as a **black box** with reduced interpretability and higher computational overhead. Aggregating 100 trees requires significantly more memory and latency to train and serve in production HR applications.
