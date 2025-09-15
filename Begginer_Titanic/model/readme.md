# Titanic Survival Prediction 🚢

This project is a classic data science challenge from Kaggle. The goal is to build a machine learning model that predicts whether a passenger on the Titanic survived or not based on their personal and travel information.

## Dataset

The dataset is provided by Kaggle and is split into two primary files:

  * `train.csv`: Contains the training data with features and the target variable (`Survived`).
  * `test.csv`: Contains the test data for which we need to predict the survival outcome.

The data includes passenger information such as age, sex, passenger class, fare, and port of embarkation.

## Project Workflow 📊

The project follows a standard machine learning workflow, from data cleaning to model evaluation.

### 1\. Data Cleaning & Preprocessing

  * **Handled Missing Values:** Imputed missing `Age` values using the median, filled missing `Embarked` values with the mode, and dropped the `Cabin` column due to a high number of missing entries.
  * **Standardized Data:** Cleaned the `Sex` column by stripping whitespace and converting to lowercase to ensure consistency.

### 2\. Feature Engineering ⚙️

  * **Title Extraction:** Created a `Title` feature by extracting titles (e.g., 'Mr', 'Mrs', 'Master') from the `Name` column and grouped rare titles into a single 'Rare' category.
  * **Family Size:** Created a `FamilySize` feature by combining the `SibSp` (siblings/spouses) and `Parch` (parents/children) columns.
  * **Age Binning:** Binned the continuous `Age` feature into categorical groups (`Child`, `Teen`, `Adult`, `Senior`) to better capture survival patterns across different age ranges.

### 3\. Exploratory Data Analysis (EDA)

  * Analyzed the relationships between various features and the `Survived` target variable.
  * Visualizations confirmed that `Sex`, `Pclass`, and `Title` were strong indicators of survival, reinforcing the "women and children first" mantra 😂 and the advantages of being in a higher passenger class.

### 4\. Data Preparation for Modeling

  * **Encoding:** Converted all categorical features (`Sex`, `Embarked`, `Title`, `AgeGroup`) into a numerical format using one-hot encoding.
  * **Scaling:** Scaled the `Fare` feature using `StandardScaler` to normalize its distribution, ensuring it didn't disproportionately influence the model.

### 5\. Modeling & Evaluation 🎯

  * **Data Splitting:** The `train.csv` data was split into a training set (80%) and a validation set (20%) to evaluate the model's performance on unseen data.
  * **Model Training:** A `RandomForestClassifier` was chosen and trained on the prepared training data.
  * **Evaluation:** The model's performance was assessed on the validation set using key classification metrics.

## Results

The final `RandomForestClassifier` model achieved an accuracy of approximately **84%** on the held-out validation set.

  * **Accuracy:** 84%
  * **Key Metrics:** The model showed a strong and balanced performance, with a precision and recall of around 0.79 and 0.81 respectively for predicting survivors.

Key features influencing survival were confirmed to be `Sex`, `Pclass`, and the engineered `Title` feature.

## Technologies Used

  * Python
  * Pandas
  * NumPy
  * Seaborn & Matplotlib
  * Scikit-learn
  * Jupyter Notebook

## How to Run

1.  Clone this repository.
2.  Install the required libraries:
    ```bash
    pip install pandas numpy seaborn matplotlib scikit-learn jupyterlab
    ```
3.  Place the `train.csv` and `test.csv` files in a `data/` directory.
4.  Open and run the Jupyter Notebook (`titanic_analysis.ipynb`) to see the full analysis and generate the submission file.