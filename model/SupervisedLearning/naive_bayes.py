import numpy as np
from sklearn.model_selection import train_test_split
from metrics.classification_metrics import accuracy_score

'''
    Naive Bayes Classifier: Phân loại dựa trên phân phối Gaussian:
        P(B|A) = (P(A|B) * P(B)) / P(A) 
    Bayesian probability: 
        posterior = (prior * likelihood) / evidence
    
    P(X|C_i) = P(x1|C_i) ... P(x_n|C_i)
    
    - Với dữ liệu liên tục, mỗi feature tuân theo Gaussian Probability:
            P(X|Y) = (1 / sqrt(2 * pi & var)) * e^(-(x - mean)^2 / (2 * var))
'''

class NaiveBayes():
    # Gaussian Naive Bayes Classifier
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y) # Classes = giá trị unique của y
        self.parameters = []

        # Calculate the mean and variance of each feature for each class
        for i, c in enumerate(self.classes):
            # Only select the sample where the label equals the given class
            X_where_c = X[np.where(y == c)]
            self.parameters.append([])
            for col in X_where_c.T:
                parameters = {'mean': np.mean(col), 'var': np.var(col)} # add mean and variance of each feature of each class
                self.parameters[i].append(parameters)

    def _calculate_likelihood(self, mean, var, x):
        # Gaussian likelihood of the data x given mean and variance
        eps = 1e-4 # Avoid division by 0
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var + eps) # 1 / sqrt(2 * pi * var)
        exponent = np.exp(-(x-mean)**2 / (2 * var + eps)) # e^(-(x - mean)^2 / (2 * var))
        return coeff * exponent # (1 / sqrt(2 * pi * var)) * e^(-(x - mean)^2 / (2 * var))


    def _calculate_prior(self, c):
        # Calculate the prior of class c (sample where class == c / total of samples) -> mean of sample has class == c
        frequency = np.mean(np.where(self.y == c)) # P(Y = c) = số sample thuộc class c / tổng sample
        return frequency

    def _classify(self, sample):
        '''
            Naive Bayes: P(Y|X) = P(X|Y) * P(Y) / P(X)
            Posterior = Likelihood * Prior / Evidence

            - P(Y|X): Posterior: Xác suất của class Y biết dữ liệu X
            - P(X|Y): Likelihood: Xác suất của dữ liệu X nếu thuộc class Y
            - P(Y): Prior: Xác suất xuất hiện của class Y
            - P(X): Evidence: Xác suất của X (xác suất của dữ liệu)

        '''
        posteriors = []
        for i, c in enumerate(self.classes):
            # Initialize posterior as prior
            posterior = self._calculate_prior(c)
            '''
                Naive assumption (independence):
                P(x1, x2, x3 | Y) = P(x1|Y) * P(x2|Y) * P(x3|Y)
                Posterior is a product of prior and likelihoods (ignoring scaling factor)
                    Posterior = prior * likelihood
            '''

            for feature_value, params in zip(sample, self.parameters[i]):
                # Likelihood of feature value given distribution of feature values given y
                likelihood = self._calculate_likelihood(params['mean'], params['var'], feature_value) # Calculate likelihood
                posterior *= likelihood
            posteriors.append(posterior)

        # Return the class with the largest posterior probability
        return self.classes[np.argmax(posteriors)]


    def predict(self, X):
        y_predict = [self._classify(sample) for sample in X]
        return y_predict

if __name__ == '__main__':
    X = np.array([
        [170, 65],
        [180, 80],
        [175, 75],
        [160, 55],
        [165, 60],
        [158, 52]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = NaiveBayes()
    model.fit(X, y)

    y_pred = model.predict(X)

    print("Prediction:", y_pred)
    print("True label:", y)

    # Accuracy
    acc = accuracy_score(y, y_pred)
    print("Accuracy:", acc)