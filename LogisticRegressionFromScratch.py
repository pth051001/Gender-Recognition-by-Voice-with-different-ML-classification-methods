import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class LogisticRegressionFromScratch:
    def __init__(self, X, y):
        # Add a column of zeros for X
        zeros_col = np.ones((X.shape[0],1))
        X = np.append(zeros_col,X,axis=1)
        # Initialize variables
        self.X = X
        self.y = y
        self.m = X.shape[0]
        self.n = X.shape[1]
        # Randomize values for theta
        self.theta = np.random.randn(X.shape[1],1)
    
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def costFunction(self):
        # Calculate predicted h then cost value
        h = self.sigmoid(np.matmul(self.X, self.theta))
        self.J = (1/self.m)*(-self.y.T.dot(np.log(h)) - (1 - self.y).T.dot(np.log(1 - h)))
        return self.J
    
    def gradientDescent(self, alpha, num_iters):
        # Keep records of cost values and thetas
        self.J_history = []
        self.theta_history = []
        for i in range (num_iters):
            # Calculate new value for h then update J_history
            h = self.sigmoid(np.matmul(self.X, self.theta))
            self.J_history.append(self.costFunction())
            self.theta_history.append(self.theta)
            self.theta = self.theta - (alpha/self.m)*(self.X.T.dot(h-self.y))
        return self.J_history, self.theta_history, self.theta
    
    def predict(self, X_test, y_test):
        # Add a column of zeros for X_test
        zeros_col = np.ones((X_test.shape[0],1))
        X_test = np.append(zeros_col, X_test, axis = 1)
        # Calculate final predicted y values after using gradient descent to update theta
        cal_sigmoid = self.sigmoid(np.matmul(X_test, self.theta))
        self.y_pred = []
        for value in cal_sigmoid:
            if value >= 0.5:
                self.y_pred.append(1)
            else:
                self.y_pred.append(0)  
        return self.y_pred
   
  
    