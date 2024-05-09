import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  balanced_accuracy_score
from sklearn.svm import SVC
from split_and_process import *
from BCR import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
x_train = pd.read_pickle('A5_2024_xtrain')
y_train = pd.read_pickle('A5_2024_ytrain')
x_test = pd.read_pickle('A5_2024_xtest')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
x_train= NormalizeData(x_train)
x_test= NormalizeData(x_test)
#x_train, pca= pca_important_features(x_train)
#x_test= pca.transform(x_test)
#apply feature selection
x_train, selctor= feature_selection(x_train, y_train, 250, mutual_info_classif)
x_test= selctor.transform(x_test)
"""#apply random forest
classifier = RandomForestClassifier(n_estimators=50,random_state=0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
#compute the BCR from BCR.py
print(balanced_accuracy_score(y_test, y_pred))
#compute the BCR from BCR.py with as argument total count and correctly classified count
cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
correctly_classified_counts = cm.diagonal()
total_counts = y_test.value_counts().reindex(classifier.classes_, fill_value=0).values
print(calculate_bcr(correctly_classified_counts, total_counts))"""
"""param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=0),
                           param_grid=param_grid,
                           scoring=balanced_accuracy_scorer,
                           cv=5,  # Number of folds in cross-validation
                           verbose=2,  # Controls the verbosity: the higher, the more messages
                           n_jobs=-1)  # Use all available cores

# Fit grid search
grid_search.fit(x_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score (balanced accuracy):", grid_search.best_score_)"""
#implement bagging over random forests
def bagging(X, y, n_estimators=10, max_samples=1.0, max_features=1.0, random_state=0):
    """
    Implement bagging over random forests.
    
    Parameters:
    - X (numpy.ndarray): The input data.
    - y (numpy.ndarray): The target labels.
    - n_estimators (int): The number of trees in the forest.
    - max_samples (float): The proportion of samples to draw from X to train each tree.
    - max_features (float): The proportion of features to draw from X to train each tree.
    - random_state (int): The seed used by the random number generator.
    
    Returns:
    - list of RandomForestClassifier: The trained trees.
    """

    trees = []
    for _ in range(n_estimators):
        # Randomly sample data with replacement
        #train_sample = train_df.sample(n=5, replace=True)

        indices = np.random.choice(len(X), size=int(max_samples * len(X)), replace=True)
        X_sampled = X[indices]        
        y_sampled = y[indices]        
        
        # Train a tree on the sampled data
        tree = RandomForestClassifier()
        tree.fit(X_sampled, y_sampled)
        trees.append(tree)
    
    return trees

#apply bagging

trees = bagging(x_train, y_train, n_estimators=100, max_samples=0.8, max_features=0.8, random_state=0)
# Predict using the bagged trees
predictions = np.array([tree.predict(x_test) for tree in trees])# Compute the majority vote
majority_vote = []
predictions=predictions.T #transpose in order to iterate through trees not samples
for pred in predictions:
    (quality, counts) = np.unique(pred, return_counts=True) #count occurences of all values
    index = np.argmax(counts)
    majority_vote.append(quality[index])
#compute the BCR from BCR.py
print(balanced_accuracy_score(y_test, majority_vote))
#compute the BCR from BCR.py with as argument total count and correctly classified count

"""param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=0),
                           param_grid=param_grid,
                           scoring=balanced_accuracy_scorer,
                           cv=5,  # Number of folds in cross-validation
                           verbose=2,  # Controls the verbosity: the higher, the more messages
                           n_jobs=-1)  # Use all available cores

# Fit grid search
grid_search.fit(x_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score (balanced accuracy):", grid_search.best_score_)
#implement bagging over random forests"""
""
    
"""# Define the parameter grid
param_grid = {
    'degree': [2, 3, 4, 5,6,7],  # Polynomial degrees to test
    'coef0': [0.2,0.5, 1.5, 2, 2.5,4, 3, 5,6,7, 9,10],   # Independent term in kernel function
    #from C =0.01 to 1 with 10 steps
    'C':[0.01,0.05,0.1, 0.15, 0.2 ,0.3, 0.4,0.6,0.7, 0.8,1, 10, 50, 100], # Regularization parameter
    'gamma': ['scale', 'auto'] , # Kernel coefficient
    
}

# Initialize the classifier
svc = SVC(kernel='poly', random_state=0)

scorer = make_scorer(balanced_accuracy_score)
# Setup the grid search with cross-validation
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring=scorer, cv=10)

# Fit the grid search to the data
grid_search.fit(x_train, y_train)

# Best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score (balanced accuracy):", grid_search.best_score_)
"""
"""
def bagging_predict(tree_list, x_test):
    
    
    predictions = np.array([tree.predict(x_test) for tree in tree_list])
    
    majority_vote = []

    predictions=predictions.T #transpose in order to iterate through trees not samples
    for pred in predictions:
        (quality, counts) = np.unique(pred, return_counts=True) #count occurences of all values
        index = np.argmax(counts)
        majority_vote.append(quality[index])        


    
    return majority_vote

#implement a bagging algorithm with n trees taking a random sample of the data
def bagging(train_df,test_df, nb_trees, seeds):
    tree_list = []
    n_samples = train_df.shape[0]
    #merge trandf and testdf together tran_df and test_df are np arrays and i want df to be a dataframe so i can use the sample method
    train_df = pd.DataFrame(train_df)
    test_df = pd.DataFrame(test_df)
    #test_df must be categorical
    test_df=pd.get_dummies(test_df)
    print(test_df)
    print(test_df)
    df = pd.concat([train_df, test_df])
    
    for i in range(nb_trees):
        train_sample = df.sample(n=n_samples, replace=True, random_state=seeds[i])
        
        X_train_sample = train_sample.iloc[:, :-3]
        y_train_sample = train_sample.iloc[:, -3]
        
        dtc = RandomForestClassifier(n_estimators=100,random_state=0)
        dtc.fit(X_train_sample, y_train_sample)
        
        tree_list.append(dtc)
    
    return tree_list
    
seeds = np.random.randint(0, 100, 100)
#apply it to the data and compute BCR
y_pred = bagging(x_train, y_train, 100, seeds)
print(balanced_accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
correctly_classified_counts = cm.diagonal()
total_counts = y_test.value_counts().reindex(np.unique(y_test), fill_value=0).values
print(calculate_bcr(correctly_classified_counts, total_counts))"""


