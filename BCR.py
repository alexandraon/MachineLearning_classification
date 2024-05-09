
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def compute_cross_val_with_BCR_sklearn(X, y, n_folds=10, classifier=SVC(kernel='linear')):
    #use cross_val_accuracy_score and balanced_accuracy_score
    cv_acc=[]
    for i in range(20):

        acc = cross_val_score(classifier, X, y, cv=n_folds, scoring='balanced_accuracy')
        cv_acc.extend(acc)
    #get confidence interval of the BCR
    #compute confidence interval
    lower = np.percentile(cv_acc, 2.5)#try with 5 percent
    upper = np.percentile(cv_acc, 97.5)
    mean= np.mean(cv_acc)
    std= np.std(cv_acc)
    return mean, (lower, upper), cv_acc, std
def compute_cross_val_one_split(X, y, classifier=SVC(kernel='linear')):
    #use balanced accuracy score
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    bcr = balanced_accuracy_score(y_test, y_pred)
    return bcr
def compute_random_sampling_bcr(X, y, iteration=100, classifier=SVC(kernel='linear')):
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
    return np.mean(bcrs), (lower, upper),  np.std(bcrs) ,bcrs

def calculate_bcr(correctly_classified_counts, total_counts):
    """
    Calculate the Balanced Classification Rate (BCR).

    Parameters:
    - correctly_classified_counts (list of int): Number of correctly classified examples for each class.
    - total_counts (list of int): Total number of examples for each class.

    Returns:
    - float: The calculated BCR value.
    """
    if len(correctly_classified_counts) != len(total_counts):
        raise ValueError("Length of correctly classified counts must match length of total counts")
    
    # Calculate the classification rate for each class and average them
    classification_rates = [tp / n if n != 0 else 0 for tp, n in zip(correctly_classified_counts, total_counts)]
    
    bcr = sum(classification_rates) / len(classification_rates)
    return bcr

#calculate BCR but in a cross validation way
def cross_validation_bcr_new(X, y, n_folds=10, classifierInit=SVC(kernel='poly', C=0.001, coef0=6, degree=8, gamma='scale', random_state=0)):
    fold_size = len(X) // n_folds
    accuracies = []
    classifier = classifierInit
    #put toghether X and y and as column name for y put label
    #X = pd.DataFrame(X)
    #y = pd.DataFrame(y)
    #y.columns = ['label']
    #X = pd.concat([X, y], axis=1)    
    for fold in range(n_folds):
        classifier = classifierInit
        start, end = fold * fold_size, (fold + 1) * fold_size
        
        X_train = np.concatenate([X[:start], X[end:]])
        y_train = np.concatenate([y[:start], y[end:]])
        X_val = X[start:end]
        y_val = y[start:end]
        
        classifier.fit(X_train, y_train)
        #accuracy is computed with BCR
        y_pred = classifier.predict(X_val)
        #print(classifier.classes_)
        #compute BCR from ,sklearn
        accuracy= balanced_accuracy_score(y_val, y_pred)
     #   print(accuracy)
        accuracies.append(accuracy)
    
    cv_acc = np.mean(accuracies)
    return cv_acc

def compute_confidence_interval_observed_and_real(X,y, n_folds=10, classifier=SVC(kernel='linear')):
    #use cross_val_accuracy_score and balanced_accuracy_score
    means=[]
    for i in range(100):
        cv_acc = cross_val_score(classifier, X, y, cv=n_folds, scoring='balanced_accuracy')
        means.append(cv_acc)
    #flatten the list and compute conficence interval
    means = np.array(means).flatten()
    observed=(np.percentile(means, 2.5), np.percentile(means, 97.5))
  
    """# Sample proportion (observed accuracy) for the  model
    p_hat_rf =  x_rf / n_rf

    # Estimate the standard error (SE) using the sample proportion
    SE_rf = np.sqrt(p_hat_rf * (1 - p_hat_rf) / n_rf)

    # Significance level for a 95% confidence interval
    alpha_rf = 0.05

    # Z-score for a 95% confidence interval
    Z_rf = norm.ppf(1 - alpha_rf / 2)

    # Confidence interval using the Normal approximation
    CI_lower_rf = p_hat_rf - Z_rf * SE_rf
    CI_upper_rf = p_hat_rf + Z_rf * SE_rf

    # Using the norm.interval function for equivalent computation
    CI_rf = norm.interval(1 - alpha_rf, p_hat_rf, SE_rf)

    print(CI_rounded_rf)"""
    #apply function above in comment for real data
    # Sample proportion (observed accuracy) for the real data
    
    return observed, means.mean()



