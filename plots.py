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
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
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
lasso_selector = SVC(kernel='linear', gamma=0.001, C=0.01)
select_from_model = SelectFromModel(lasso_selector, threshold='1.4*mean')

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

def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training set size (sample)")
    plt.ylabel("BCR Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig("Learning_curve.pdf")
    return plt

#plot_learning_curve(pipeline, "Learning Curve", x_train, y_train, cv=10, n_jobs=1)

#plot confidence interval of 95 percent of values of BCR with x the BCR score and y the number of samples you can use the function under
"""def compute_random_sampling_bcr(X, y, iteration=100, classifier=SVC(kernel='linear')):
    bcrs=[]
    for i in range(iteration):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        bcr = balanced_accuracy_score(y_test, y_pred)
        print(f"Iteration {i+1}: {bcr}")
        bcrs.append(bcr)
    lower= np.percentile(bcrs, 2.5)
    upper= np.percentile(bcrs, 97.5)
    return np.mean(bcrs), (lower, upper),  np.std(bcrs) ,bcrs"""
def plot_confidence_interval(X,y, iterations):
    mean, (lower, upper), std, bcrs = compute_random_sampling_bcr(X, y, iterations,pipeline)
    plt.errorbar(range(iterations), mean, yerr=[mean-lower, upper-mean], fmt='o')
    plt.xlabel("Number of samples")
    plt.ylabel("BCR")
    plt.title("Confidence Interval of BCR for 95 percent of values over {iterations} iterations")   
    plt.savefig("Confidence_interval.pdf")
    return plt

plot_confidence_interval(x_train, y_train, 10)
