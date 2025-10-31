import matplotlib.pyplot as plt
import numpy as np

# Function that returns fitted model parameters to the dataset at datapath for each choice in degrees.

def main(datapath, degrees):
    # Input
    # --------------------
    # datapath : A string specifying a .txt file 
    # degrees : A list of positive integers.
    #    
    # Output
    # --------------------
    # paramFits : a list with the same length as degrees, where paramFits[i] is the list of
    #             coefficients when fitting a polynomial of d = degrees[i].
    
    paramFits = []
    with open(datapath, 'r') as file:
        data = file.readlines()
    x = []
    y = []
    for line in data:
        [i, j] = line.split()
        x.append(float(i))
        y.append(float(j))
        
    # iterate through each n in the list degrees, calling the feature_matrix and least_squares functions to solve
    # for the model parameters in each case. Append the result to paramFits each time.

    for d in degrees:
        X = feature_matrix(x, d)
        B = least_squares(X, y)
        paramFits.append(B)

    return paramFits

# Function that returns the feature matrix for fitting a polynomial of degree d based on the explanatory variable
# samples in x.

def feature_matrix(x, d):
    # Input
    # --------------------
    # x: A list of the independent variable samples
    # d: An integer
    #
    # Output
    # --------------------
    # X : A list of features for each sample, where X[i][j] corresponds to the jth coefficient
    #     for the ith sample. Viewed as a matrix, X should have dimension (samples, d+1).
    # There are several ways to write this function. The most efficient would be a nested list comprehension
    # which for each sample in x calculates x^d, x^(d-1), ..., x^0.
    # Please be aware of which matrix colum corresponds to which degree polynomial when completing the writeup.
    X = []
    for sample in x:
        row = []
        for power in range(d, -1, -1):
            row.append(sample ** power)
        X.append(row)
    return X

# Function that returns the least squares solution based on the feature matrix X and corresponding target variable samples in y.
def least_squares(X, y):
    # Input
    # --------------------
    # X : A list of features for each sample
    # y : a list of target variable samples.
    # Outut
    # --------------------
    # B : a list of the fitted model parameters based on the least squares solution.
    
    X = np.array(X)
    y = np.array(y)
    # Use the matrix algebra functions in numpy to solve the least squares equations. This can be done in just one line.
    B, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    return B


if __name__ == "__main__":
    datapath = "poly.txt"

    file = open(datapath, 'r')
    data = file.readlines()
    x = []
    y = []
    for line in data:
        [i, j] = line.split()
        x.append(float(i))
        y.append(float(j))

    ### Part 1 ###

    ### Part 2 ###
    # The paramater values for degrees 2 and 4 have been provided as test cases in the README.
    # The output should match up to at least 3 decimal places rounded
    # Write out the resulting estimated functions for each d.
    degrees = [1, 2, 3, 4, 5, 6] #
    paramFits = main(datapath, degrees)

    print("=== POLYNOMIAL COEFFICIENTS ===")
    for idx, deg in enumerate(degrees):
        print(f"y_hat(x_{deg})")
        print(paramFits[idx])
        print("****************")

    ### Part 3 ###
    # Use the 'scatter' and 'plot' functions in the `matplotlib.pyplot` module.
    # Draw a scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, color='black', label='data', s=20, alpha=0.7)
    x_sorted = sorted(x)
    x_dense = np.linspace(min(x), max(x), 300)

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']


    for i, params in enumerate(paramFits):
        y_pred = []
        for x_val in x_dense:
            y_val = 0
            for j, coeff in enumerate(params):
                power = len(params) - j - 1
                y_val += coeff * (x_val ** power)
            y_pred.append(y_val)

        plt.plot(x_dense, y_pred,
                 color=colors[i],
                 label=f'd={degrees[i]}',
                 linewidth=2)

    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.title('Polynomial Regression Fits of Various Degrees', fontsize=14)
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(5)

    ### Part 4 ###
    # Use the degree that best matches the data as determined in Problem 3 above.
    
    best_degree = 3
    X_test = feature_matrix([2], best_degree)
    prediction = X_test[0] @ paramFits[degrees.index(best_degree)]

    print(f"\n=== PREDICTION ===")
    print(f"Best fitting polynomial degree: {best_degree}")
    print(f"Predicted value for x=2: {prediction:.5f}")

    # Print the actual equation
    coefficients = paramFits[degrees.index(best_degree)]
    print(f"Equation: y = ", end="")
    terms = []
    for j, coeff in enumerate(coefficients):
        power = len(coefficients) - j - 1
        if power == 0:
            terms.append(f"{coeff:.3f}")
        elif power == 1:
            terms.append(f"{coeff:.3f}x")
        else:
            terms.append(f"{coeff:.3f}x^{power}")
    print(" + ".join(terms))
