import numpy as np

def generate_dataset(num_points, variance, step=1, correlation=True):
    X = [i for i in range(0, num_points)]
    if correlation is True:
        steps = step * np.array(X)
        Y = steps + np.random.normal(1, np.sqrt(variance), num_points)
    else:
        Y = np.random.normal(1, np.sqrt(variance), num_points)
    return np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64)

def generate_testset(X, Y, variance, set_size = 0.2, step=1, correlation=True):
    # match test variance to the main dataset variance
    # set_size is proportional to main dataset size (i.e., 0.2 = 20%)
    X_test = [i for i in range(len(X), int(np.ceil(len(X)+set_size*len(X))))]
    if correlation is True:
        steps = step * np.array(X_test)
        Y_test = steps + np.random.normal(1, np.sqrt(variance), len(X_test))
    else:
        Y_test = np.random.normal(1, np.sqrt(variance), len(X_test))
    return np.array(X_test), np.array(Y_test)