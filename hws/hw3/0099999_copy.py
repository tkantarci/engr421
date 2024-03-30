import math
import matplotlib.pyplot as plt
import numpy as np

# read data into memory
data_set_train = np.genfromtxt("hw03_data_set_train.csv", delimiter = ",", skip_header = 1)
data_set_test = np.genfromtxt("hw03_data_set_test.csv", delimiter = ",", skip_header = 1)

# get x and y values
x_train = data_set_train[:, 0]
y_train = data_set_train[:, 1]
x_test = data_set_test[:, 0]
y_test = data_set_test[:, 1]

# set drawing parameters
minimum_value = 1.6
maximum_value = 5.1
x_interval = np.arange(start = minimum_value, stop = maximum_value, step = 0.001)

def plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat):
    fig = plt.figure(figsize = (8, 4))
    plt.plot(x_train, y_train, "b.", markersize = 10)
    plt.plot(x_test, y_test, "r.", markersize = 10)
    plt.plot(x_interval, y_interval_hat, "k-")
    plt.xlim([1.55, 5.15])
    plt.xlabel("Time (sec)")
    plt.ylabel("Signal (millivolt)")
    plt.legend(["training", "test"])
    plt.show()
    return(fig)



# STEP 3
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def regressogram(x_query, x_train, y_train, left_borders, right_borders):
    # your implementation starts below
    elems_in_bins = np.asarray([np.sum((left_borders[b] < x_train) & (x_train <= right_borders[b]))
                    for b in range(len(left_borders))])
    sum_in_bins = np.asarray([np.sum(y_train * ((left_borders[b] < x_train) & (x_train <= right_borders[b]))) for b in range(len(left_borders))])

    r = sum_in_bins / elems_in_bins
    
    y_hat =[np.multiply(((left_borders[b] <= x_query) & (x_query < right_borders[b])), r[b]) for b in range(len(r))] 
    y_hat = np.asarray(y_hat)
    y_hat = np.sum(y_hat, axis=0)
    # your implementation ends above
    return(y_hat)
    
bin_width = 0.35
left_borders = np.arange(start = minimum_value, stop = maximum_value, step = bin_width)
right_borders = np.arange(start = minimum_value + bin_width, stop = maximum_value + bin_width, step = bin_width)

y_interval_hat = regressogram(x_interval, x_train, y_train, left_borders, right_borders)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("regressogram.pdf", bbox_inches = "tight")

y_test_hat = regressogram(x_test, x_train, y_train, left_borders, right_borders)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Regressogram => RMSE is {} when h is {}".format(rmse, bin_width))



# STEP 4
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def running_mean_smoother(x_query, x_train, y_train, bin_width):
    # your implementation starts below
    def w(x_query, x_train, h):
        w_matrix = [(((x - x_train) / h) >= -1/2) & (((x - x_train) / h) < 1/2) for x in x_query]
        w_matrix = np.asarray(w_matrix)
        return w_matrix

    w_matrix = w(x_query, x_train, bin_width)
    w_sums = np.sum(w_matrix, axis=1)

    y_hat_matrix = w_matrix * y_train
    y_hat_sums = np.sum(y_hat_matrix, axis=1)

    y_hat = y_hat_sums / w_sums
    # your implementation ends above
    return(y_hat)

bin_width = 0.35

y_interval_hat = running_mean_smoother(x_interval, x_train, y_train, bin_width)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("running_mean_smoother.pdf", bbox_inches = "tight")

y_test_hat = running_mean_smoother(x_test, x_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Running Mean Smoother => RMSE is {} when h is {}".format(rmse, bin_width))



# STEP 5
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def kernel_smoother(x_query, x_train, y_train, bin_width):
    # your implementation starts below
    def k(x_query, x_train, h):
        u = np.asarray([((x - x_train) / h) for x in x_query])
        kernel_matrix = ((1 / np.sqrt(2 * math.pi)) * np.exp((-u ** 2) / 2))
        return kernel_matrix

    kernel_matrix = k(x_query, x_train, bin_width)
    kernel_sums = np.sum(kernel_matrix, axis=1)

    y_hat_matrix = kernel_matrix * y_train
    y_hat_sums = np.sum(y_hat_matrix, axis=1)

    y_hat = y_hat_sums / kernel_sums
    # your implementation ends above
    return(y_hat)

bin_width = 0.35

y_interval_hat = kernel_smoother(x_interval, x_train, y_train, bin_width)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("kernel_smoother.pdf", bbox_inches = "tight")

y_test_hat = kernel_smoother(x_test, x_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Kernel Smoother => RMSE is {} when h is {}".format(rmse, bin_width))
