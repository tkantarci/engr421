import numpy as np
import pandas as pd



X = np.genfromtxt("hw01_data_points.csv", delimiter = ",", dtype = str)
y = np.genfromtxt("hw01_class_labels.csv", delimiter = ",", dtype = int)



# STEP 3
# first 50000 data points should be included to train
# remaining 44727 data points should be included to test
# should return X_train, y_train, X_test, and y_test
def train_test_split(X, y):
    # your implementation starts below
    X_train = X[:50000]
    y_train = y[:50000]
    
    X_test = X[50000:]
    y_test = y[50000:]
    # your implementation ends above
    return(X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# STEP 4
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    class_priors = [np.mean(y == c) for c in (1, 2)]
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 5
# assuming that there are K classes and D features
# should return four numpy arrays with shape (K, D)
def estimate_nucleotide_probabilities(X, y):
    # your implementation starts below
    pAcd = [np.average(X == "A", axis=0, weights=(y == c)) for c in (1, 2)]
    pCcd = [np.average(X == "C", axis=0, weights=(y == c)) for c in (1, 2)]
    pGcd = [np.average(X == "G", axis=0, weights=(y == c)) for c in (1, 2)]
    pTcd = [np.average(X == "T", axis=0, weights=(y == c)) for c in (1, 2)]
    # your implementation ends above
    return(pAcd, pCcd, pGcd, pTcd)

pAcd, pCcd, pGcd, pTcd = estimate_nucleotide_probabilities(X_train, y_train)
print(pAcd)
print(pCcd)
print(pGcd)
print(pTcd)



# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, pAcd, pCcd, pGcd, pTcd, class_priors):
    # your implementation starts below
    score_values = np.zeros(2 * len(X)).reshape(len(X), 2)
    for c in range(len(class_priors)):
        for i in range(len(X)):
            sum = 0
            for d in range(len(X[0])):
                if (X[i][d] == "A"):
                    sum += np.log(pAcd[c - 1][d])
                if (X[i][d] == "C"):
                    sum += np.log(pCcd[c - 1][d])
                if (X[i][d] == "G"): 
                    sum += np.log(pGcd[c - 1][d])
                if (X[i][d] == "T"):
                    sum += np.log(pTcd[c - 1][d])
            score_values[i][c - 1] = sum + np.log(class_priors[c - 1]) 
    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_test)



# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    predicted_vals = np.zeros(len(scores))
    
    for i, score in enumerate(scores):
        if (score[0] > score[1]):
            predicted_vals[i] = 1
        else:
            predicted_vals[i] = 2
    
    confusion_matrix = pd.crosstab(predicted_vals.T, y_truth.T)
    confusion_matrix = confusion_matrix.to_numpy()
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)
