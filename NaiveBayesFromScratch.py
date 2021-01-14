import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class NaiveBayesFromScratch():
    def __init__(self, X, y):
        self.num_examples, self.num_features = X.shape
        self.num_classes = len(np.unique(y))

    def fit(self, X, y):
        self.classes_mean = {}
        self.classes_variance = {}
        self.classes_prior = {}

        for c in range(self.num_classes):
            X_c = X[y == c]

            self.classes_mean[str(c)] = np.mean(X_c, axis=0)
            self.classes_variance[str(c)] = np.var(X_c, axis=0)
            self.classes_prior[str(c)] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        probs = np.zeros((X.shape[0], self.num_classes))
        for c in range(self.num_classes):
            prior = self.classes_prior[str(c)]
            probs_c = multivariate_normal.pdf(X, mean=self.classes_mean[str(c)], cov=self.classes_variance[str(c)])
            probs[:,c] = probs_c*prior
        return np.argmax(probs, 1)