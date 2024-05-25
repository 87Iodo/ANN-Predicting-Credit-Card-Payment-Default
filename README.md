# Credit Card Payment Default Prediction

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Tools](#tools)
- [Data Cleaning/Preparation](#data-cleaningpreparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Building and Evaluation](#model-building-and-evaluation)
- [Results/Findings](#resultsfindings)
- [Recommendations](#recommendations)
- [Limitations](#limitations)
- [References](#references)

### Project Overview

This project aims to create a model in python to predict if customers will default in their next month credit card payment. The model can help credit card company to allocate appropriate resources to assit customer that has difficult making credit card payment, thereby improving receiving and cash flow.

### Data Sources

Taiwan Credit Card Payment Data: The dataset that is used to construct the model is "CC_Default.csv". It is a subset of original data containing customer information and their 6 months credit card payment details. 
Source: Lichman, M. (2013). UCI Machine Learning Repository https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients. Irvine, CA: University of California, School of Information and Computer Science.

### Tools

- Jupyter Notebook
  - Numpy
  - Pandas
  - Matplotlib
  - Seaborn
  - Statsmodels
  - SciPy
  - scikit-learn

### Data Cleaning/Preparation

In preparing the data, following tasks are performed:
1. Data loading.
2. Inspection of data including data type and missing data.

### Exploratory Data Analysis

EDA involves finding relation of features to default payment in next month:
1. Plot univariate distribution.
2. Feature engineering: Summarizing repayment status of all 6 months and create new feature PAY_AGG. Finding linear relationship in default payment rate of each PAY_AGG group
3. Categorizing amount in bill statement and amount paid by segregating data into bins with different width, thereby creating better distribution profile. Finding linear relationship in default payment rate of each group of amount in bill statement and amount paid.
4. Finding relationship between age and default payment.
5. Removing unknown from category features (education, marriage).
6. Finding relation ship of categorical features to target variables by using mosaic graph and chi-aquare test.
7. Drop all irrelavent features including ID.
8. Check multicollinearity by ploting correlation heatmap and VIF.
9. Remove features that are highly correlated.

### Model Building and Evaluation

Categorical features (sex, marriage, education) are encoded with one hot encoding to prevent ordinal effect.
Imbalance data is processed with SMOTE and then normalized.

Different machine learning models including logistic regression, random forest, and SVM are created and evaluated with confusion matrix.
ANN model is also created with dropout and earlystopping in training.

### Results/Findings

The models are compared based on the evaluation metrics. The Random Forest model has the lowest accuracy (0.74) and highest F1-score (0.51), indicating even though overall correctness of the model is slightly lower, it has better mean of precision and recall. The Random Forest model also has lowest precision (0.44) but almost two times higher recall (0.61) for the "Default" class compared to other model, suggesting it is best at correctly identifying defaults from actual defaults. A higher rate of false negatives is acceptable in this case as the actions based on the prediction (e.g. sending payment reminder to customer) is not likely to cause costly consequences.

Given a much better recall and comparably similat accuracy, precision as well as f1 score, the Random Forest model appears to be the best choice among the options to achieve the objective of the model.
Besides that, Ramdom Forest requires less computational power and is more interpretable than other models. Its superior performance on the specific key metrics justifies its use in this context.

### Recommendations

The models are struggling with minority class. Training models with full dataset instead of a subset is recommended.
Hyperparameters of the models should be optimized to get best result.

### Limitations

Some research on domain knowledge is required to properly interpret the data.

### References
- [UCI](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
