import numpy as np
from model.DeepLearning.activation_functions import Sigmoid
from utils.data_manipulation import make_diagonal

'''
    Logistic Regression: Classification model using sigmoid to classify
    - regression architecture: z = X * w
    - sigmoid: sigmoid(z) = 1 / (1 + e^(-z)) <-> 1 / (1 + np.exp(-z)) -> 0 < sigmoid(z) < 1
    - prediction: y_pred = sigmoid(X * w)
    - Loss function: Binary Cross Entropy Loss:
        BCE = -[y * log(y_pred) + (1 - y) * log(1 - y_pred)]
            = - (1 / N) * Xichma(yi * log(pi) + (1 - yi) * log(1 - pi))
    - Multiclass Cross Entropy Loss:
        CE = - (1 / N) * Xichma_N * Xichma_C (y_i,j * log(p_i,j)) 
        
    - Gradient: Gradient(w) = X.T * (y_pred - y)
                w = w - learning_rate * X.T * (y_pred - y)
'''
class LogisticRegression():
    def __init__(self, n_iterations=4000, learning_rate=1e-3, gradient_descent=True):
        self.param = None
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.sigmoid = Sigmoid()

    def _initialize_parameters(self, X):
        # initialize random parameters same as regression
        n_features = X.shape[1]
        limit = 1 / np.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y):
        # Initialize weight
        self._initialize_parameters(X)

        for i in range(self.n_iterations):
            # Calculate prediction by sigmoid
            y_pred = self.sigmoid(X.dot(self.param)) # sigmoid(X * w))
            if self.gradient_descent:
                # param <-> w
                self.param -= self.learning_rate * -(y - y_pred).dot(X) # w = w - learning_rate * X.T * (y_pred - y) = w - learning_rate * -(y - y_pred) * X
            else:
                '''
                    Newton Method | Newton-Raphson
                    - Gradient(sigmoid(z)) = sigmoid(z) * (1 - sigmoid(z))
                    - Tạo ma trận đường chéo diag D dựa trên gradient sigmoid (X * w)
                    - Update Newton: 
                        w = (X.T * D * X)^-1 * X.T * (D * X * w + y - y_pred)
                    - Ma trận nghịch đảo ảo -> pinv <-> []^-1
                '''
                diag_gradient = make_diagonal(self.sigmoid.gradient(X.dot(self.param))) # gradient(sigmoid(X * w))
                self.param = np.linalg.pinv(X.T.dot(diag_gradient).dot(X)).dot(X.T).dot(diag_gradient.dot(X).dot(self.param) + y - y_pred)

    def predict(self, X):
        # Làm tròn prediction sigmoid(X * w))
        y_pred = np.round(self.sigmoid(X.dot(self.param))).astype(int)
        return y_pred