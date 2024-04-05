import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats

group_means = np.array([[-6.0, -1.0],
                        [-3.0, +2.0],
                        [+3.0, +2.0],
                        [+6.0, -1.0]])

group_covariances = np.array([[[+0.4, +0.0],
                               [+0.0, +4.0]],
                              [[+2.4, -2.0],
                               [-2.0, +2.4]],
                              [[+2.4, +2.0],
                               [+2.0, +2.4]],
                              [[+0.4, +0.0],
                               [+0.0, +4.0]]])

# read data into memory
data_set = np.genfromtxt("hw05_data_set.csv", delimiter = ",")

# get X values
X = data_set[:, [0, 1]]

# set number of clusters
K = 4

# STEP 2
# should return initial parameter estimates
# as described in the homework description
def initialize_parameters(X, K):
    # your implementation starts below
    means = np.genfromtxt("hw05_initial_centroids.csv", delimiter = ",")
    D = dt.cdist(means, X)
    memberships = np.argmin(D, axis=0)
    priors = np.asarray([np.mean(memberships==k) for k in range(K)])
    covariances = np.asarray([np.cov(X[memberships == k].T) for k in range(K)])
    # your implementation ends above
    return(means, covariances, priors)

means, covariances, priors = initialize_parameters(X, K)

# STEP 3
# should return final parameter estimates of
# EM clustering algorithm
def em_clustering_algorithm(X, K, means, covariances, priors):
    # your implementation starts below
    def update_H(X, H, means, covariances, K):
        N = X[:, 0].size
        for i in range(N):
            for k in range(K):
                mixt_density = stats.multivariate_normal.pdf(X[i] , means[k].T, covariances[k])
                H[i][k] = mixt_density * priors[k]
        
                denom = np.sum([stats.multivariate_normal.pdf(X[i] , means[c].T, covariances[c]) * priors[c] for c in range(K)])
                H[i][k] /= denom
    
        return H
    
    def update_priors(priors, H, K):
        N = X[:, 0].size
        priors = np.asarray([np.sum(H[:, k]) / N for k in range(K)])
        return priors
    
    def update_means(means, H, X, K):
        N = X[:, 0].size
        for k in range(K):
            means[k] = np.sum([H[i, k] * X[i] for i in range(N)], axis=0) / np.sum(H[:, k])
        return means
    
    def update_covariances(covariances, means, H, X, K):
        N = X[:, 0].size
        for k in range(K):
            covariances[k] = np.sum([H[i, k] * np.matmul((X[i] - means[k])[:, None], (X[i] - means[k])[:, None].T) for i in range(N)], axis=0) / np.sum(H[:, k])
        return covariances
    
    def iteration(means, covariances, priors, H, X, K):
        N = X[:, 0].size
    
        # E Step
        H = update_H(X, H, means, covariances, K)
    
        # M Step
        priors = update_priors(priors, H, K)
        means = update_means(means, H, X, K)
        covariances = update_covariances(covariances, means, H, X, K)
    
        return (means, covariances, priors, H)

    # Initialize H matrix
    H = np.empty((X[:, 0].size, K))
    
    iter_count = 100
    for i in range(iter_count):
        means, covariances, priors, H = iteration(means, covariances, priors, H, X, K)

    assignments = np.argmax(H, axis=1)
    # your implementation ends above
    return(means, covariances, priors, assignments)

means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)

# STEP 4
# should draw EM clustering results as described
# in the homework description
def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):
    # your implementation starts below
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00"])
    
    fig = plt.figure(figsize=(8, 8))
    for c in range(K):
        plt.plot(X[assignments == c, 0], X[assignments == c, 1], ".", markersize = 10,
                         color = cluster_colors[c])
    
    x1_interval = np.arange(-8, +8, 0.01)
    x2_interval = np.arange(-8, +8, 0.01)
    x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
    coords = np.empty(x1_grid.shape + (2,))                
    coords[:, :, 0] = x1_grid
    coords[:, :, 1] = x2_grid 
    
    for c in range(K):
        density = stats.multivariate_normal.pdf(coords,group_means[c], group_covariances[c])
        plt.contour(x1_grid, x2_grid,density, 1, colors="#000000",levels=[0.01] ,linestyles='dashed') 
    
        predicted_density = stats.multivariate_normal.pdf(coords,np.reshape(means[c], (2,)), covariances[c])
        plt.contour(x1_grid, x2_grid,predicted_density, 1, colors=cluster_colors[c],levels=[0.01]) 

    plt.show()
    fig.savefig("fig.pdf", bbox_inches = "tight")
    # your implementation ends above
    
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)