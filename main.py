import numpy as np
import matplotlib.pyplot as plt
import dataGenerator as data

def fit_line(X, Y):
    """
    Determines the best linear model to fit the given 2-dimensional dataset.
    Returns the best fit line, the slope, the y-intercept, and the
    coefficient of determination.
    """
    m = ((np.mean(X) * np.mean(Y) - np.mean(X*Y)) /
         ((np.mean(X))**2 - np.mean(X**2)))
    b = np.mean(Y) - m*(np.mean(X))
    y_cap = [(m*x)+b for x in X]
    SS_y_cap = sum((y-y_cap)**2)
    SS_y_mean = sum([(y-np.mean(Y))**2 for y in Y])
    coeff_of_deter = 1 - (SS_y_cap/SS_y_mean)
    return y_cap, m, b, coeff_of_deter

def LinearRegression(X, Y, X_test, Y_test=None):
    """"
    Fit a Linear Regression model to X and Y sets, and predict on X_test.
    Returns fitted model,  predictions, computed average error, and fitness score(r2).
    """
    model, slope, y_int, fitness_score = fit_line(X, Y)
    predictions = np.array([(slope*x)+y_int for x in X_test])
    if Y_test is not None and len(Y_test) == len(X_test):
        sum_of_sqrs = sum((predictions - Y_test)**2)
        mean_error = np.sqrt(sum_of_sqrs/len(Y_test))
    else:
        mean_error = None
    return model, predictions, mean_error, fitness_score



# Generate training data
x, y = data.generate_dataset(num_points=50, variance=20)

# Generate test data
x_test, y_test = data.generate_testset(x, y, variance=20, set_size=0.2)

# Fit model to training data & make predictions on test data
model, predictions, mean_error, fitness_score = LinearRegression(x, y, x_test, y_test)

# Visualize results
plt.scatter(x,y, color='black')
plt.scatter(x_test, y_test, color='red')
plt.plot(x, model, color='black')
plt.plot(x_test, predictions, color='green')
plt.show()
print("Model fitness score(r^2):", fitness_score)
print("Average error:", mean_error)
print("Predictions:", predictions)