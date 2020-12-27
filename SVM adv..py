import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split


class SVM:
    def hinge_loss(self, w, x_, y_):
        loss = 0
        v = y_ * np.dot(w, x_.T)
        loss += max(0, 1 - v[-1])
        return loss

    def fit(self, X, y, lr=0.001, n_iters=1000, lambda_param=0.01):
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(X.shape[3])
        for _ in range(n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) ) >= 1
                if condition:
                    self.w -= lr * (2 * lambda_param * self.w)
                else:
                    self.w -= lr * (2 * lambda_param * self.w - np.dot(x_i, y_[idx]))

    def predict(self, X):
        approx = np.dot(X, self.w)
        pred = np.sign(approx)
        pred = np.where(pred == -1, 0, 1)
        return pred

    def score(self, x, y):
        pred = self.predict(x)
        return 1 - np.sum(np.abs((pred - y))) / len(y)


if __name__ == "__main__":

    df = pd.read_csv(r'iris.csv')

    df = df.drop('Id', axis=1)

    df['Species'] = [1 if x == 'Iris-setosa' else 0 for x in df['Species']]

    X = df[['SepalLengthCm', 'SepalWidthCm']]

    X = X.values

    y = df['Species']

    y = y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = SVM()

    model.fit(X_train, y_train)

    print(model.score(X_test, y_test))