import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  balanced_accuracy_score
from sklearn.svm import SVC
from split_and_process import *
from BCR import *
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
x_train = pd.read_pickle('A5_2024_xtrain')
y_train = pd.read_pickle('A5_2024_ytrain')
x_test = pd.read_pickle('A5_2024_xtest')
x_train= NormalizeData(x_train)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.10, random_state=1)

y_train= NormalizeLabels(y_train)
y_test= NormalizeLabels(y_test)
print(x_train.shape)
print(x_test.shape)

#x_train, pca= pca_important_features(x_train, 0.94)
#x_test= pca.transform(x_test)
# Pipeline for feature selection
pipe_fs = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('scaler', StandardScaler()),  # Standardizing data for better SVM performance
    ('svc', SVC(kernel='poly', random_state=0))
])

# Pipeline for PCA
pipe_pca = Pipeline([
    ('pca', PCA()),
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='poly', random_state=0))
])
# Parameters grid for PCA
param_grid_pca = {
    'pca__n_components': [0.90,0.93, 0.95],
    'svc__C': [0.001, 0.01, 0.1, 1, 10, 100]
}

# Scorer for GridSearch
scorer = make_scorer(balanced_accuracy_score)

param_grid_fs = {
    'feature_selection__k': [170, 200, 230],  # Number of features
    'svc__C': [0.001, 0.01, 0.05,  0.1, 0.15, 1, 10, 50, 100],
    'svc__coef0': [1.5, 2,  2.5, 3, 4, 5, 6, 7,  10],
    'svc__degree': [2, 3, 4, 5, 6, 7, 8],
    'svc__gamma': ['scale', 'auto']
}

# Parameters grid for PCA
param_grid_pca = {
    'pca__n_components': [ 0.90,0.93, 0.95],
    'svc__C': [0.001, 0.01, 0.05,  0.1, 0.15, 1, 10, 50, 100],
    'svc__coef0': [1.5, 2,  2.5, 3, 4, 5, 6, 7,  10],
    'svc__degree': [2, 3, 4, 5, 6, 7, 8],
    'svc__gamma': ['scale', 'auto']
}
"""grid_fs = GridSearchCV(pipe_fs, param_grid_fs, scoring=scorer, cv=5, verbose=2)
grid_fs.fit(x_train, y_train)
print("Best parameters (Feature Selection):", grid_fs.best_params_)
print("Best cross-validation score (Feature Selection):", grid_fs.best_score_)"""

# PCA grid search
grid_pca = GridSearchCV(pipe_pca, param_grid_pca, scoring=scorer, cv=5, verbose=2)
grid_pca.fit(x_train, y_train)
print("Best parameters (PCA):", grid_pca.best_params_)
print("Best cross-validation score (PCA):", grid_pca.best_score_)
#x_train,features=applyRFE(x_train, y_train, 1000)
#x_train, selctor= feature_selection(x_train, y_train, 230, mutual_info_classif)
#RFE_selector = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=0), n_features_to_select=250, step=1)
#add non gene columns to the selected columns x_train
#x_train = pd.concat([pd.DataFrame(x_train), pd.DataFrame(non_gene_column_name_values)], axis=1)
#x_train= np.array(x_train)
#x_test= selctor.transform(x_test)
#x_train, pca= pca_important_features(x_train, 0.94)
#x_test= pca.transform(x_test)
#sF=225
#cv=5
#Best parameters: {'C': 0.001, 'coef0': 6, 'degree': 8, 'gamma': 'scale'}
#create svc with these params
#svc = GradientBoostingClassifier(n_estimators=100, random_state=0)
#apply linear regression
#svc = SVC(kernel='linear')
#compute the BCR with cross validation
"""
y_pred = svc.fit(x_train, y_train).predict(x_test)
print(balanced_accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
correctly_classified_counts = cm.diagonal()
total_counts = y_test.value_counts().reindex(np.unique(y_test), fill_value=0).values
print(calculate_bcr(correctly_classified_counts, total_counts))"""
#compute the BCR with cross validation
#print(cross_validation_bcr_new(x_train, y_train, 10, svc))