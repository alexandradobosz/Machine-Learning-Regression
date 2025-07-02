# Machine-Learning-Regression
This is the project that I have completed during my second semester. My task was to apply various machine learning algorithms to build a model explaining the prices of apartments
based on the training sample and generate predictionsfor all observations from the test sample. I was using training and test data:
apartments_train.csv - that contains 156454 observations and 34 columns along with the target variable price_z.
apartments_test.csv – test data containing 39114 observations and 33 columns without the target variable.

To predict the apartment prices prize_z, three algorithms were implemented. In each of them the apartment_train was
divided into three sets - training, validation and test (60-20-20).
  
LINEAR REGRESSION - Enhanced with interaction effects. Carried out using a structured pipeline (PolynomialFeatures(), StandardScaler()), 5-fold cross-validation was applied

RIDGE REGRESSION - Structured with pipeline with PolynomialFeatures and StandardScaler components. RandomizedSearchCV was applied with
5-fold cross-validation. Following Ridge parameters were tuned: alpha - logarithmically 0.01 to 100, solver - set to auto and
lsqr, 15 combinations.

LASSO REGRESSION - Pipeline containing PolynomialFeatures and StandardScaler as well as Lasso(max_iter=5000). Hyperparameter tunning via
RandomizedSearchCV used with 5-fold cross-validation. model__alpha set from 0.01 to 10 on a logarithmic scale, search limited to 10 iterations.

After each of the identification of best model - three performance metrics were computed:

Cross-validated RMSE on training set: Averaged across 5 folds, this metric measures in-sample predictive accuracy while accounting for variability across folds.

Validation RMSE: Measures how well the model generalizes to new, but related, data (from the validation set).
Test RMSE: The ultimate evaluation metric, reflecting model performance on unseen, real-world-like data.

Selected algorithm - Linear Regression
Cross-validated RMSE on the training set: 96,807.51 - this value reflects the average RMSE obtained across five cross-validation folds within the training data.

Validation RMSE: 97,304.67 - The RMSE calculated on the validation set (previously unseen during model fitting) closely aligns with the cross-validated RMSE.
This proximity suggests that the model generalizes well within the internal data split and has not been overly tuned to the
training folds.

Final Test RMSE: 97,935.09 - The final performance on the external test set shows only a slight increase in RMSE relative to the training and validation metrics.

In the context of predicting the apartment prices Linear Regression emerged as the most suitable model - as
evidence by its lowest Root Mean Squared Error. It achieved the strongest predictive performance - in
comparison to more complex and regularised models like Ridge and Lasso (as well as very time consuming
Random Forest). Interaction term has helped to capture important relationships without adding complexity.
Additionally, I think that the transparency and interpretability make it a good choice for the real estate
analysis. The well structure data and preprocessing allow this simple model to outperform. It was also the
most computationally efficient model - I think it is a practical and reliable choice for this predictive task.
