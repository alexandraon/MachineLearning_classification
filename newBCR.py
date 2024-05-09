from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import pandas as pd
from sklearn.linear_model import LassoCV
import numpy as np
x_train = pd.read_pickle('A5_2024_xtrain')
y_train = pd.read_pickle('A5_2024_ytrain')
x_test = pd.read_pickle('A5_2024_xtest')

y_train = y_train.replace('ER+/HER2-', 0)
y_train = y_train.replace('ER-/HER2-', 1)
y_train = y_train.replace('HER2+', 2)
class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for column in X.columns:
            if column in ['event', 'node']:
                # Ensure the columns contain numeric data before conversion
                X[column] = pd.to_numeric(X[column], errors='coerce').astype(int)     
            elif column == 'grade':
                X = pd.get_dummies(X, columns=[column])
                #print dumies columns
                #print mean of column grade_0
                #print(X['grade_0'].mean())
                # Convert all grade columns to integer type in one go
                grade_cols = [col for col in X.columns if col.startswith('grade_')]
                X[grade_cols] = X[grade_cols].astype(int)
                #ensure you have the same columns as during fit
                
            else:
                # Reshape data for scaling
                X[column] = self.scaler.fit_transform(X[[column]])  # Directly use double brackets for correct shape
        print(X.columns)
        return X
    
    
class CustomFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold='mean'):
        self.threshold = threshold
        self.estimator = LassoCV(random_state=0, tol=0.01)
        self.selector = None
        self.gene_cols = None
        self.other_cols = None

    def fit(self, X, y=None):
        # Separate gene columns and other columns
        self.gene_cols = [col for col in X.columns if col.startswith('gene_')]
        self.other_cols = [col for col in X.columns if col not in self.gene_cols]

        # Apply feature selection only to gene columns
        self.estimator.fit(X[self.gene_cols], y)
        self.selector = SelectFromModel(self.estimator, prefit=True, threshold=self.threshold)
        return self

    def transform(self, X, y=None):
        # Apply feature selection to gene columns
        X_gene_new = self.selector.transform(X[self.gene_cols])

        # Always include other columns
        X_other = X[self.other_cols]

        # Concatenate selected gene columns and other columns
        X_new = np.concatenate([X_gene_new, X_other], axis=1)

        print(f"Number of features selected: {X_new.shape[1]}")
        return X_new
    
class DimensionChecker(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        if X.shape[1] != self.n_features_:
            #print the differnt features between the two dataframes
            print(X.columns)
       
            raise ValueError(f"Number of features of the input must be equal to the number of features of the data passed to fit. Got {X.shape[1]} features, expected {self.n_features_}.")
        return X
# Create a pipeline that first applies the custom scaler then applies the classifier
classifier=LogisticRegression(multi_class='multinomial' ,solver='lbfgs', max_iter=1000,tol=1e-1)


pipeline = Pipeline([
    ('scaler', CustomScaler()),
    ('checker', DimensionChecker()),
    ('selector', CustomFeatureSelector()),
    ('classifier', classifier)
])

# Apply cross-validation on the pipeline
scores = cross_val_score(pipeline, x_train, y_train, cv=10, scoring='balanced_accuracy')

print(f"Cross-validation scores: {scores}")
print(f"Average cross-validation score: {scores.mean()}")