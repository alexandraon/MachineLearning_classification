import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, balanced_accuracy_score

# Load data
x_train = pd.read_pickle('A5_2024_xtrain')
y_train = pd.read_pickle('A5_2024_ytrain')
x_test = pd.read_pickle('A5_2024_xtest')
# Custom transformer for data normalization with consistent feature order
class CustomNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, initial_columns=None):
        self.scaler = StandardScaler()
        self.initial_columns = initial_columns

    def fit(self, X, y=None):
        # Capture the initial order of columns when fitting to ensure consistent order in transform
        if 'grade' in X.columns:
            # Generate all potential grade categories from the training data
            temp_X = pd.get_dummies(X, columns=['grade'])
            self.initial_columns = temp_X.columns
        else:
            self.initial_columns = X.columns
        return self

    def transform(self, X):
        if 'event' in X.columns and 'node' in X.columns:
            X.loc[:, ['event', 'node']] = X[['event', 'node']].apply(pd.to_numeric, errors='coerce').astype(int)
        if 'grade' in X.columns:
            X = pd.get_dummies(X, columns=['grade'])
            # Efficiently add missing columns using concat and convert set to list
            missing_cols = set(self.initial_columns) - set(X.columns)
            if missing_cols:
                missing_data = pd.DataFrame(0, index=X.index, columns=list(missing_cols))
                X = pd.concat([X, missing_data], axis=1)
        
        #get  columns other than event node and grades apply the scaler
        other_cols = [col for col in X.columns if col not in ['event', 'node'] and not col.startswith('grade_')]
        X[other_cols] = self.scaler.fit_transform(X[other_cols])

        # Reorder columns to match the initial training order
        X = X.reindex(columns=self.initial_columns)
        return X


# Define label encoding function
def normalize_labels(y):
    label_mapping = {'ER+/HER2-': 0, 'ER-/HER2-': 1, 'HER2+': 2}
    y = y.replace(label_mapping)
    return y

y_train = normalize_labels(y_train)

# Feature selection using Lasso
lasso_selector = LassoCV(cv=10, tol=0.1)

#lasso_selector = SVC(kernel='linear')
select_from_model = SelectFromModel(lasso_selector)

# Classifier
#C': 1, 'fit_intercept': True, 'max_iter': 1000, 'penalty': 'l1', 'solver': 'saga', 'tol': 0.1
logistic= LogisticRegression(multi_class='multinomial', solver='saga', max_iter=1000, tol=1e-1, penalty='l1', C=1, fit_intercept=True)

# Initialize and fit the normalizer to get initial column order from training data
normalizer = CustomNormalizer()
normalizer.fit(x_train)

# Create the pipeline
pipeline = Pipeline([
    ('normalizer', normalizer),  # Normalization with consistent order
    ('feature_selection', select_from_model),
    ('classifier', logistic)
])

# Parameter grid for GridSearchCV
param_grid = {
    'feature_selection__estimator__tol': [1e-1,1],  # Lasso regularization parameter
     'feature_selection__threshold': ['mean', 'median', '1.5*mean'],
     # Logistic regression regularization parameter
    'classifier__C': [0.1,1,10],
    'classifier__fit_intercept': [True, False],
    'classifier__max_iter': [1000],
    'classifier__penalty': ['l1', 'l2', 'none  '],
    'classifier__solver': ['saga', 'lgfbs'],
}

# Scorer
scorer = make_scorer(balanced_accuracy_score)

# GridSearchCV to optimize the pipeline
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=scorer, verbose=2    )

# Fit and evaluate the model
grid_search.fit(x_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")

# Save the best model
best_model = grid_search.best_estimator_
best_model.fit(x_train, y_train)
#save in predictions.csv the predictiosn for xtest
y_pred = best_model.predict(x_test)
#rechange the 0 1 and 2 to the labels
y_pred = pd.Series(y_pred).replace({0: 'ER+/HER2-', 1: 'ER-/HER2-', 2: 'HER2+'})
y_pred.to_csv('predictions.csv', index=False)

#print percentage of each class and their name in the predisctions 
print(y_pred.value_counts(normalize=True))

