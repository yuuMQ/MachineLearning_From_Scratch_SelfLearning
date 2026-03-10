import numpy as np
from metrics.regression_metrics import mean_squared_error, mean_absolute_error

'''
    Regularization: L1 (Lasso) và L2 (Ridge), giảm overfitting bằng cách phạt các weight lớn trong model
        LOSS = Loss_data + Loss_regularization
    
    - L1: Loss_l1 = alpha * xichma(|w_i|) = alpha * ||w|| (Norm) | ||w|| = |w1| + |w2| + .... + |wn|
    - L2: Loss_l2 = (1/2) * alpha  * xichma((w_i)^2) = alpha * 0.5 * (wT * w) | wT * w = (w1)^2 + .... + (wn)^2
    
'''

class l1_regularization():
    # Lasso
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.linalg.norm(w, ord=1) # linalg.norm(): Linear Algebra, norm tính độ lớn của vector | linalg.norm(w) = Xichma(|wi|)

    def grad(self, w):
        return self.alpha * np.sign(w) # np.sign(w) lấy dấu của |w| | grad_w (|w|) = sign(w)


class l2_regularization():
    # Ridge
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * 0.5 * w.T.dot(w) # w.T.dot(w) là tích ma trận w^2 tương tự như công thức wi^2

    def grad(self, w):
        return self.alpha * w


class l1_l2_regularization():
    # Regularization for Elastic Net Regression
    def __init__(self, alpha, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        l1_contr = self.l1_ratio * np.linalg.norm(w, ord=1) # _contr: contribution
        l2_contr = (1 - self.l1_ratio) * 0.5 * w.T.dot(w)
        return self.alpha * (l1_contr + l2_contr)

    def grad(self, w):
        l1_contr = self.l1_ratio * np.sign(w)
        l2_contr = (1 - self.l1_ratio) * w
        return self.alpha * (l1_contr + l2_contr)


class Regression(object):
    '''
        Base Regression model | y = ax + b = w1*X + w0
        -> y_pred = w0 + w1*x1 + w2*x2 + ... + w_n*x_n
        -> y_pred = X*w
            + X: ma trận dữ liệu
            + w: weight (vector trọng số)
        - Base Regression model này là bộ khung cho các model regression khác, sử dụng gradient descent.
    '''
    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initialize_weights(self, n_features):
        # Initialize weights randomly [-1/N, 1/N]
        limit = 1 / np.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, )) # Khởi tạo w ngẫu nhiên, với n features thì sẽ có n giá trị của w. w là array list của các weight được khởi tạo ngẫu nhiên w[w1, w2..., w_n].

    def fit(self, X, y):
        '''
            Thêm cột giá trị 1 vào X -> bias:
                Ex: X[[2], [3], [4]] -> X[[1, 2], [1, 3], [1, 4]]
            - X = [1 x], w = [w0, w1]
            -> y_pred = w * X <-> 1 * w0 + w1 * X
        '''
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = [] # Lưu loss
        self.initialize_weights(n_features=X.shape[1])

        # Gradient Descent
        for i in range(self.n_iterations):
            y_pred = X.dot(self.w) # Tính y_pred = X * w

            # Calculate l2 regularization
            mse = mean_squared_error(y, y_pred, self.regularization(self.w)) # Tính loss (sử dụng mean squared error)
            self.training_errors.append(mse)

            # Gradient descent of l2 loss
            grad_w = -(y - y_pred).dot(X) + self.regularization.grad(self.w) # Tính gradient descent bằng cách đạo hàm loss

            # Update weight
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        # Insert constant ones for bias weights | Thêm cột bias vào X
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w) # Thực hiện dự đoán y_predict
        return y_pred

class LinearRegression(Regression):
    def __init__(self, n_iterations=100, learning_rate=1e-3, gradient_descent=True):
        super(LinearRegression, self).__init__(n_iterations=n_iterations, learning_rate=learning_rate)
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent

        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0

    def fit(self, X, y):
        '''
            If not gradient descent -> Least squares approximation of w
            Loss: L(w) = (y - y_pred)^2 = (y - Xw)^2
            -> Gradient: L(w) = -2 * X.T * (y - Xw) = -2 * X.T * y + 2 * X.T * Xw = 0
            -> w = (X.T * y) / (X.T * X) = (X.T * Xw)^-1 * X.T * y
            - 1 số vấn đề với least squares khi XT * X không khả nghịch -> inverse không tồn tại
            -> Moore-penrose Pseudoinverse X+ (không hiểu cái này lắm, đại loại là giải quyết vấn đề khả nghịch)
            - w = X+ * y
            -> X+ = (X.T * X) ^ -1 * X.T

            - Tính pseudoinverse bằng SVD (Singular Value Decomposition)
                X = U * S * V.T -> X+ = U * S+ * V.T
                - X (m x n)

                Matrix              Ý nghĩa
                U: m x m        orthogonal matrix (ma trận trực giao (left singular vectors))
                S: m x n        singular values
                V: n x n        orthogonal matrix (ma trận trực giao (right singular vectors))
                trong đó S+ là nghịch đảo các singular values

            - Moore-penrose Pseudoinverse làm ngườ lại SVD: V * S * U.T
        '''
        if not self.gradient_descent:
            # insert constant ones for bias weights
            X = np.insert(X, 0, 1, axis=1)

            # Calculate weights by lease squares (using Moore-Penrose pseudoinverse)
            U, S, V = np.linalg.svd(X.T.dot(X)) # X.T * X = U * S * V.T
            S = np.diag(S) # Tạo ma trận đường chéo

            # np.linalg.pinv(S): tạo ma trận nghịch đảo giả của S. pinv: pseudo-inverse
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T) # (X.T * X)+ = V * S+ * U.T
            self.w = X_sq_reg_inv.dot(X.T).dot(y) # w = (X.T * X)+ * (X.T * y)

        else:
            super(LinearRegression, self).fit(X, y)


if __name__ == '__main__':
    X = np.array([
        [1],
        [2],
        [3],
        [4],
        [5]
    ])
    y = np.array([3, 5, 7, 9, 11])

    model = LinearRegression(n_iterations=100, learning_rate=1e-3, gradient_descent=True)
    model.fit(X, y)
    print(model.w)
    predictions = model.predict(X)
    print(predictions)