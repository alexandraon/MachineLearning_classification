import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
#scaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,RFE, SelectFromModel
from BCR import *
#import lassoCV
from sklearn.linear_model import LassoCV
x_train = pd.read_pickle('A5_2024_xtrain')
y_train = pd.read_pickle('A5_2024_ytrain')
x_test = pd.read_pickle('A5_2024_xtest')
#import mutual_info_classif
from sklearn.feature_selection import mutual_info_classif

#print into a file the the descirbe of the data that ir readable bc csvv is not readable
#print(x_train.describe().to_csv('x_train_describe.csv'))
#decribe in csv of event column 
#x_train['event'].describe().to_csv('x_train_event_describe.csv')
#decribe in csv of node column 
#x_train['node'].describe().to_csv('x_train_node_describe.csv')


#normalize numerical values in x_train
def NormalizeData(X):
    """
    Normalize data in dataframe X based on column names:
    - Columns 'event' and 'node' are cast to integers (ensure they contain numeric data).
    - Column 'grade' is one-hot encoded.
    - All other columns are scaled using StandardScaler.
    """
    scaler = StandardScaler()  # Initialize scaler once

    for column in X.columns:
        if column in ['event', 'node']:
            # Ensure the columns contain numeric data before conversion
            X[column] = pd.to_numeric(X[column], errors='coerce').astype(int)     
        elif column == 'grade':
            X = pd.get_dummies(X, columns=[column])
            #there are 4 columns grade_0, grade_1, grade_2, grade_3
            # Convert all grade columns to integer type in one go
            grade_cols = [col for col in X.columns if col.startswith('grade_')]
            X[grade_cols] = X[grade_cols].astype(int)

        else:
            # Reshape data for scaling
            X[column] = scaler.fit_transform(X[[column]])  # Directly use double brackets for correct shape
    return X
    
def NormalizeLabels(y):
    """
    put ints instead of labels in y, put 0 1 or 2 instead of ER+/HER2-, ER-/HER2-, HER2+.
    """
    y = y.replace('ER+/HER2-', 0)
    y = y.replace('ER-/HER2-', 1)
    y = y.replace('HER2+', 2)
    return y
def feature_selection(X, y, k=100, score_func='f_classif'):
    """
    Select the k best features from X based on the target labels y.
    """
    #extract non gene columns
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X, y)
    X = selector.transform(X)
    #implement RFE to get the best features
    

    
    return X, selector
def featre_selection_transform(X, selector):
    """
    Select the k best features from X based on the target labels y.
    """
    X = selector.transform(X)
    return X
def pca_important_features(df, variance_threshold=0.95):
    """
    Apply PCA to X and keep the columns names that explain 95% of the variance and return the columns names that are kept

    """

    #apply pca to keep 0;95 of the variance
    
    pca = PCA(n_components=variance_threshold)

    # Fit PCA on the normalized data
    pca.fit(df)

    # Transform the data according to the PCA
    reduced_features = pca.transform(df)

    # The number of components that explain up to 95% of the variance
    n_components = pca.n_components_

    print(f"Number of components that explain {variance_threshold}% of the variance: {n_components}")
    #print("Explained variance by each component:", pca.explained_variance_ratio_)
    #get the columns names that are kept not the n first components
    return reduced_features, pca

def pca_transform(df, pca):
    """
    Apply PCA to X and keep the columns names that explain 95% of the variance and return the columns names that are kept

    """
    
    # Transform the data according to the PCA
    reduced_features = pca.transform(df)

    # The number of components that explain up to 95% of the variance
    n_components = pca.n_components_

    print(f"Number of components that explain 95% of the variance: {n_components}")
    #print("Explained variance by each component:", pca.explained_variance_ratio_)
    #get the columns names that are kept not the n first components
    return reduced_features
    
def applyLassoCv(X, y, alpha=0.001):
    #apply only on genes (begin with G_ ) and keep other columns
    X = X.loc[:, X.columns.str.startswith('G_')]
    #apply lasso cv
    lassocv = LassoCV(alphas=[alpha,0.1, 0.01])
    lassocv.fit(X, y.values.ravel())

    # Get indices of non-zero coefficients
    non_zero_indices = np.where(lassocv.coef_ != 0)[0]
    print(f"Number of features with non-zero coefficients: {len(non_zero_indices)}")

    # Filter features to keep only those with non-zero coefficients
    filtered_features = X.iloc[:, non_zero_indices]
    #add columns that are not genes
    filtered_features = pd.concat([filtered_features, X.loc[:, ~X.columns.str.startswith('G_')]], axis=1)
    return filtered_features
    
def neighborhood_mutual_information(X, y, n_neighbors=30):
    n_samples = len(X)
    mi_scores = np.zeros(X.shape[1])
    # Calculate MI for randomly selected neighborhoods
    for _ in range(n_neighbors):
        indices = np.random.choice(n_samples, size=n_neighbors, replace=False)
        #X is np array so iloc is not working
        local_X = X[indices]
        #does this take the values as
        local_y = y[indices]
        
        # Compute mutual information for the subset
        mi_scores += mutual_info_classif(local_X, local_y)

    # Average the MI scores over all neighborhoods
    mi_scores /= n_neighbors

    return mi_scores

def neighborhood_mi_scores(X, y):
    scores = neighborhood_mutual_information(X, y)
    p_values = np.ones_like(scores)  # Dummy p-values since MI does not provide these
    return scores, p_values

def applyRFE(X, y, n_features_to_select=1000, step=1):
    """
    Apply Recursive Feature Elimination (RFE) to select the best features from X based on the target labels y.
    """
    # Initialize the RFE object
    estimator = RandomForestClassifier(n_estimators=100, random_state=0)
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
    X = selector.fit_transform(X, y)

    return X

def applySelectFromModel(X, y, threshold='mean'):
    """
    Apply SelectFromModel to select the best features from X based on the target labels y.
    """
    # Initialize the SelectFromModel object
    #estimator has a parameter penalty set to l1
    #estimator = RandomForestClassifier(n_estimators=100, random_state=0)
    #increas regularization in lasso cv to get less features

    estimator = LassoCV(cv=10, random_state=0, tol=0.1).fit(X, y)
    selector = SelectFromModel(estimator, prefit=True)
    X = selector.fit_transform(X, y)
    #print number of features selected
    print(f"Number of features selected: {X.shape[1]}")
    return X, selector