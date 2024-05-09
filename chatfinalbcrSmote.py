import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline  
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, balanced_accuracy_score
from imblearn.over_sampling import SMOTE  # Import SMOTE
from BCR import compute_cross_val_with_BCR_sklearn, compute_random_sampling_bcr, compute_cross_val_one_split

# Load data
x_train = pd.read_pickle('A5_2024_xtrain')
y_train = pd.read_pickle('A5_2024_ytrain')
x_test = pd.read_pickle('A5_2024_xtest')

# Custom transformer for data normalization with consistent feature order
class CustomNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, initial_columns=None):
        self.scaler = StandardScaler()
        self.initial_columns = initial_columns
        self.scaler_time = None
        self.scaler_sizeTum = None
        self.scaler_age = None

    def fit(self, X, y=None):
        if 'grade' in X.columns:
            temp_X = pd.get_dummies(X, columns=['grade'])
            self.initial_columns = temp_X.columns
        else:
            self.initial_columns = X.columns
        return self

    def transform(self, X):
        if 'event' in X.columns and 'node' in X.columns:
            X.loc[:, ['event', 'node']] = X[['event', 'node']].apply(pd.to_numeric, errors='coerce').astype(int)
        if 'time' in X.columns:
            self.scaler_time = StandardScaler()
            X['time'] = self.scaler_time.fit_transform(X[['time']])
        if 'sizeTum' in X.columns:
            self.scaler_sizeTum = StandardScaler()
            X['sizeTum'] = self.scaler_sizeTum.fit_transform(X[['sizeTum']])
        if 'age' in X.columns:
            self.scaler_age = StandardScaler()
            X['age'] = self.scaler_age.fit_transform(X[['age']])
        if 'grade' in X.columns:
            X = pd.get_dummies(X, columns=['grade'])
            missing_cols = set(self.initial_columns) - set(X.columns)
            if missing_cols:
                missing_data = pd.DataFrame(0, index=X.index, columns=list(missing_cols))
                X = pd.concat([X, missing_data], axis=1)
        other_cols = [col for col in X.columns if col not in ['event', 'node'] and not col.startswith('grade_')]
        X[other_cols] = self.scaler.fit_transform(X[other_cols])
        X = X.reindex(columns=self.initial_columns)
        return X

# Define label encoding function
def normalize_labels(y):
    label_mapping = {'ER+/HER2-': 0, 'ER-/HER2-': 1, 'HER2+': 2}
    y = y.replace(label_mapping)
    return y

y_train = normalize_labels(y_train)

# Feature selection using Lasso
selector = SVC(kernel='linear', gamma=0.001, C=0.01)
select_from_model = SelectFromModel(selector, threshold='1.4*mean')

# Classifier
logistic = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=1000, tol=1e-1, penalty='l1', C=0.5, fit_intercept=True)

# Initialize and fit the normalizer to get initial column order from training data
normalizer = CustomNormalizer()
normalizer.fit(x_train)

# Create the pipeline with SMOTE
pipeline = Pipeline([
    ('normalizer', normalizer),
    ('sampling', SMOTE()),  # SMOTE applied after normalization
    ('feature_selection', select_from_model),
    ('classifier', logistic)
])

# Compute new BCR
print("Computing new BCR")
print(compute_random_sampling_bcr(x_train, y_train, 20, pipeline))
#print(compute_cross_val_with_BCR_sklearn(x_train, y_train, 20, pipeline))
# Predict on x_test
y_pred = pipeline.fit(x_train, y_train).predict(x_test)

# Rechange the 0, 1, and 2 to the labels
y_pred = pd.Series(y_pred).replace({0: 'ER+/HER2-', 1: 'ER-/HER2-', 2: 'HER2+'})
y_pred.to_csv('predictions.csv')

print("Predictions saved to predictions.csv")
print(y_pred.value_counts())
print(y_pred.value_counts(normalize=True))
