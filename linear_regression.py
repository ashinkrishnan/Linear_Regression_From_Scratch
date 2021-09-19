import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated


# Testing
if __name__ == "__main__":
#MSE
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
#R2score
    def r2_score(y_true, y_pred):
        Nr = np.sum(np.square(y_true - y_pred))
        Dr = np.sum(np.square(y_true-np.mean(y_true)))

        R_sqr = 1-(Nr/Dr)
        return R_sqr


    import matplotlib.pyplot as plt
    #from sklearn.model_selection import train_test_split
    #from sklearn import datasets

    import pandas as pd
    df = pd.read_csv('Salary_data.csv')

    y_train = df.pop("Salary")
    X_train = df
    # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42)

    # X,y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    # print(X_train.shape)
    # print(y_train.shape)

    regressor = LinearRegression(learning_rate=0.01, n_iters=1000)
    regressor.fit(X_train, y_train)

    n = int(input("enter the length of the test data frame"))
    inst = []
    lab = []

    for i in range(n):
        inst_ = float(input("enter the {} out of {} sample(YearsExperience)".format(i+1,n)))
        lab_ = float(input("enter the {} out of {} label(Salary)".format(i+1,n)))

        inst.append(inst_)
        lab.append(lab_)

    df_dict ={
        "YearsExperience" :inst,
        "Salary" : lab
    }

    df_test = pd.DataFrame(df_dict)
    print(df_test)

    y_test = df_test.pop("Salary")
    X_test = df_test


    predictions = regressor.predict(X_test)

    
    mse = mean_squared_error(y_test, predictions)
    print("MSE:", mse)

    accu = r2_score(y_test, predictions)
    print("Accuracy:", accu)

    print("predictions:",predictions)

    y_pred_line = regressor.predict(X_test)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X_test, y_pred_line, color="black",linewidth=2, label="Prediction")
    plt.show()
