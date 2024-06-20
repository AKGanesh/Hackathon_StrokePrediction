![Logo](https://github.com/AKGanesh/Hackathon_StrokePrediction/blob/main/heart.jpg)

# Hackathon - Stroke Prediction

The goal of this competition is to figure out if a person will experience a stroke based on their age, nature of work, urban/rural residence, marital status, and several clinical levels.

Make sure to focus on each part of the Machine Learning pipeline to submit the best possible solution you can.

You will be allowed to submit at max 10 solutions per day, and your overall top 2 shall be considered for the final leaderboard.

## Implementation Details

- Dataset: Please check the Dataset details below
- Model: Logistic Regression, SVM, Random Forest
- Input: Please check the columns below
- Output: Prone to Stroke (0 or 1)
- Scores : Accuracy and f1
- Others : How to deal with imbalanced data, hyperparameter tuning, selecting the features, best model and SMOTEENN etc.,

## Dataset details

Files:

- stroke_train_set_dirty.csv - the training set with input features and GT value for 'stroke'
- stroke_test_set_nogt.csv - the test set with only input features to evaluate your model
- sample_submission.csv - a sample submission file in the expected format

Columns:

- gender: "Male", "Female" or "Other"
- age: age of the patient
- hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
- heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
- ever_married: "No" or "Yes"
- work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
- Residence_type: "Rural" or "Urban"
- avg_glucose_level: average glucose level in blood
- bmi: body mass index
- smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"\*
- stroke: 1 if the patient had a stroke or 0 if not (Prediction)

## Process

- Data and Pre-processing
  - Import the required libraries
  - Read the dataset
  - Process the data and clean (Imputation), Drop/Fill the NaNs, substitute with ffill/bfill, mean, median, avg, mode etc.,
  - Drop the duplicates
  - Drop any columns that are not required
  - Categorical to Numeric (One hot encoding)
  - EDA (Explore the relations with Seaborn/Matplotlib)
  - Check the Correlation with heatmap
- Model Development
  - Divide dataframe to Input and Output (X,y)
  - Feature Selection (SelectKBest ex:)
  - Data Normalization (MinMaxScalar ex:)
  - Choose Model, Fit and Predict
  - Check the scores (accuracy/f1/r2 ex:)
  - Hyperparameter tuning (GridSearchCV ex:)
  - Get the best params and fit to the model
- Test and Predict
  - Test against the ground truth to check the model perf
  - Incase of Imabalanced datasets, Apart from choosing the right algorithm and weightage, it is important to work on the dataset with SMOTEENN, SMOTE, NearMiss from https://imbalanced-learn.org/

## Evaluation and Results

| Method                   | Accuracy Score | f1Score |
| ------------------------ | -------------- | ------- |
| Logistic Regression      | 0.79           | 0.82    |
| SVM                      | 0.81           | 0.83    |
| Random Forest Classifier | 0.99           | 0.99    |

## Observations

- I have tried with Logistic Regression, SVM and Random Forest Classifier. After using the techniques for imbalanced data, All started giving good outcome. Random Forest performed better than the others in this situation.
- Rely on f1score than accuracy score, its been observed initially even though the accuracy score is high the outcome is bad. RFC performed better than SVM(weight=Balanced). Feature selection is also playing a key role on the outcome, choose that technique wisely.
- Finally, I tried to apply Hyperparameter tuning using GridSearchCV, this also has an impact on the final outcome in a positive way.

## Libraries

**Language:** Python

**Packages:** Sklearn, Numpy, Pandas, Seaborn, Matplotlib, imbalanced-learn

## Roadmap

- Research on techniques for Classification Problems, especially for multi-class and multi-label

- Work on various other methods for under/over sampling

## FAQ

#### Whats is F1 Score?

F1-score is a metric used to evaluate the performance of a model in machine learning tasks, particularly those involving classification. It's especially useful when dealing with imbalanced datasets.
If your dataset has a lot more examples of one class compared to another, F1-score is a better choice than accuracy, which can be misleading in such cases.

The F1-score ranges from 0 to 1, with 1 being the best score (perfect precision and recall). A score closer to 0 indicates poor performance.

#### What is hyperparameter tuning?

Hyperparameter tuning is a crucial step in the machine learning workflow. It helps you fine-tune your model and unlock its full potential.
Imagine hyperparameters as the dials on a machine learning model. They influence how the model learns from data and makes predictions. This can significantly impact the accuracy and generalizability of your model.

#### What is a classification problem?

A classification problem in machine learning is a task where the goal is to predict the category (or class) that a new data point belongs to. It's essentially about sorting things into predefined groups. Different types include Binary, Multi-Class and Multi-Label.

#### What is Data Imputation?

Data imputation is a technique used in statistics to address missing data in datasets. It's essentially the process of filling in those missing values with substituted values.

#### What is EDA?

EDA stands for Exploratory Data Analysis. It's a crucial first step in data science projects. It involves using various techniques to get a feel for your data and uncover interesting patterns, trends, or relationships. We can make use of Vizualization libraries for this.

## Acknowledgements

- https://scikit-learn.org/
- https://www.kaggle.com/abbhinavvenkat
- https://imbalanced-learn.org/

## Contact

For any queries, please send an email (id on github profile)

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
