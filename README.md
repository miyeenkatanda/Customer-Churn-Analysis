# Customer Churn Analysis

## Project Overview

This project focuses on analyzing customer churn for a telecommunications company. The primary goal is to identify key factors that contribute to customer churn and to build predictive models to forecast whether a customer is likely to churn. By understanding the drivers of churn, the company can implement targeted retention strategies to reduce customer attrition and improve overall business performance.

The analysis involves several stages, including data loading, cleaning, exploratory data analysis (EDA), feature engineering, model building, and evaluation. Various machine learning algorithms are explored to find the best performing model for churn prediction. Additionally, model interpretability techniques like LIME and SHAP are used to understand the predictions.

## Data Source

The dataset used for this analysis is the "Telco Customer Churn" dataset, which is publicly available. It was loaded from the following URL:
`https://zhang-datasets.s3.us-east-2.amazonaws.com/telcoChurn.csv`

The dataset contains customer-level information, including:
-   **Customer Demographics:** gender, SeniorCitizen, Partner, Dependents
-   **Account Information:** tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
-   **Services Subscribed:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
-   **Churn Status:** Churn (Yes/No) - This is the target variable.

## Libraries Used

The project utilizes a range of Python libraries for data manipulation, visualization, machine learning, and model interpretation:

-   **Data Handling & Manipulation:**
    -   `pandas`
    -   `numpy`
-   **Data Visualization:**
    -   `matplotlib.pyplot`
    -   `seaborn`
-   **Machine Learning & Model Evaluation:**
    -   `scikit-learn`:
        -   `ensemble` (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)
        -   `linear_model` (LogisticRegression)
        -   `tree` (DecisionTreeClassifier)
        -   `model_selection` (train_test_split, cross_val_score)
        -   `metrics` (confusion_matrix, accuracy_score, ConfusionMatrixDisplay, roc_auc_score, precision_score, f1_score, recall_score, roc_curve, auc)
        -   `preprocessing` (LabelEncoder)
    -   `xgboost` (xgb)
    -   `tensorflow.keras` (Sequential, Dense, Dropout, EarlyStopping)
-   **Model Interpretability:**
    -   `lime` (lime_tabular)
    -   `shap`
-   **Hyperparameter Optimization:**
    -   `optuna`
-   **Others:**
    -   `warnings` (to filter warnings)

## Methodology

### 1. Data Loading and Initial Exploration
-   The dataset was loaded into a pandas DataFrame.
-   Initial exploration included:
    -   Displaying the head of the DataFrame (`df.head()`).
    -   Getting information about data types and non-null counts (`df.info()`).
    -   Generating descriptive statistics for numerical features (`df.describe()`).
    -   Checking for missing values (`df.isnull().sum()`).
    -   Examining data types of columns (`df.dtypes`).

### 2. Data Cleaning and Preprocessing
-   Column names were stripped of leading/trailing whitespace.
-   The `TotalCharges` column, initially an object type, was converted to a numeric type. Rows with missing `TotalCharges` (after coercion) were dropped.
-   The `tenure` column was also converted to a numeric type.

### 3. Exploratory Data Analysis (EDA)
-   **Target Variable Distribution:** The distribution of the 'Churn' variable was visualized using a count plot to understand the class balance.
-   **Categorical Features Exploration:** Count plots were generated for each categorical feature, showing the distribution of churn within each category. This helped identify potential relationships between categorical predictors and churn.
-   **Numerical Feature Exploration:** Histograms and Kernel Density Estimate (KDE) plots were used to visualize the distribution of numerical features (SeniorCitizen, tenure, MonthlyCharges, TotalCharges) with respect to churn.

### 4. Feature Engineering (Implicit)
-   While not explicitly detailed as a separate "Feature Engineering" section in the notebook, the conversion of `TotalCharges` and `tenure` to numeric types is a form of preprocessing.
-   Categorical features were likely encoded (e.g., using LabelEncoder or OneHotEncoder, though the exact method for all features isn't fully detailed in the initial EDA cells) before model training. The notebook imports `LabelEncoder`.

### 5. Model Building and Training
The notebook imports several classifiers, indicating that multiple models were likely trained and evaluated:
-   Logistic Regression
-   Decision Tree Classifier
-   Random Forest Classifier
-   Gradient Boosting Classifier
-   AdaBoost Classifier
-   XGBoost Classifier
-   Neural Network (using TensorFlow/Keras)

The data was split into training and testing sets (`train_test_split`).

### 6. Model Evaluation
Various metrics were used for model evaluation, imported from `sklearn.metrics`:
-   Confusion Matrix
-   Accuracy Score
-   ROC AUC Score
-   Precision Score
-   F1 Score
-   Recall Score
-   ROC Curve and AUC

### 7. Model Interpretation
-   **LIME (Local Interpretable Model-agnostic Explanations):** Used to explain individual predictions.
-   **SHAP (SHapley Additive exPlanations):** Used for global and local feature importance, including bar plots and summary plots.

### 8. Hyperparameter Optimization
-   `optuna` was imported, suggesting that hyperparameter tuning was performed for one or more models to optimize their performance.

## Key Findings (Inferred from EDA and Model Interpretability Plots)

While specific model performance metrics and detailed conclusions are typically found at the end of a notebook (which might not be fully represented in the provided snippets), the EDA and model interpretability plots likely revealed insights such as:

-   Customers with **month-to-month contracts** are more likely to churn.
-   Customers with **higher monthly charges** might have a higher churn rate, especially if not coupled with long tenure or valuable services.
-   **Lower tenure** is generally associated with higher churn.
-   Specific services like **OnlineSecurity** and **TechSupport** (when absent) might increase churn probability.
-   **Payment methods** like Electronic check might be associated with higher churn.
-   Features like `Contract`, `tenure`, `OnlineSecurity`, `TechSupport`, and `MonthlyCharges` are likely to be important predictors of churn, as indicated by the SHAP plots.

## How to Use

1.  **Prerequisites:** Ensure you have Python installed along with the libraries listed in the "Libraries Used" section. You can typically install them using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow lime shap optuna
    ```
2.  **Dataset:** Download the `telcoChurn.csv` dataset or ensure the notebook can access it from the provided URL.
3.  **Run the Notebook:** Execute the cells in the Jupyter Notebook (`Customer Churn Analysis.ipynb`) sequentially to reproduce the analysis.

## Future Work (Potential)

-   Explore more advanced feature engineering techniques.
-   Experiment with other machine learning models or ensemble methods.
-   Conduct more in-depth hyperparameter tuning for all models.
-   Deploy the best performing model as an API for real-time churn prediction.
-   Develop a dashboard to visualize churn trends and model insights for business stakeholders.
-   Investigate the impact of customer service interactions on churn.



something
