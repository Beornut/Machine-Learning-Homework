import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


class LRModel:
    def __init__(self, use_gradient_descent=True, max_iter_step=100000, learning_rate=1e-4):
        self.use_gradient_descent = use_gradient_descent
        self.max_iter_step = max_iter_step
        self.learning_rate = learning_rate
        self.w = []

    def __gradient_decent(self, dataset):
        x = np.c_[np.zeros((dataset.shape[0], 1)), dataset[dataset.columns[:-1]]]
        y = dataset[dataset.columns[-1]]
        w = np.zeros(x.shape[1])
        n = x.shape[0]
        min_loss = float('inf')
        best_w = []
        for i in range(self.max_iter_step):
            dif = np.matmul(x, w) - y
            loss = np.sum(dif**2) / n
            if loss < min_loss:
                min_loss = loss
                best_w = w
            w = w - self.learning_rate * np.matmul(x.T, dif) / n
        return best_w

    def __least_square(self, dataset):
        x = np.c_[np.zeros((dataset.shape[0], 1)), dataset[dataset.columns[:-1]]]
        y = dataset[dataset.columns[-1]]
        return np.matmul(np.linalg.inv(np.matmul(x.T, x) + np.eye(x.shape[1]) * self.learning_rate), np.matmul(x.T, y))

    def fit(self, dataset):
        if self.use_gradient_descent:
            self.w = self.__gradient_decent(dataset)
        else:
            self.w = self.__least_square(dataset)

    def predict(self, dataset):
        x = np.c_[np.zeros((dataset.shape[0], 1)), dataset[dataset.columns[:-1]]]
        y_ = np.matmul(x, self.w)
        y = dataset[dataset.columns[-1]]
        y_mean = np.mean(y)
        n = dataset.shape[0]
        rmse = np.sqrt(np.sum((y - y_)**2) / n)
        r_square = 1 - np.sum((y - y_)**2) / np.sum((y - y_mean)**2)
        return rmse, r_square


def load_insurance_data(path):
    insurance = pd.read_csv(path)
    insurance['sex'] = insurance['sex'].map({'male': 1, 'female': 0})
    insurance['smoker'] = insurance['smoker'].map({'yes': 1, 'no': 0})
    insurance = pd.get_dummies(data=insurance, columns=['region'], prefix=['region'])
    label = insurance['charges']
    insurance.drop('charges', axis=1, inplace=True)
    insurance.insert(insurance.columns.size, 'charges', label)
    train_dataset = insurance.sample(frac=0.7, random_state=0, axis=0)
    test_dataset = insurance[~insurance.index.isin(train_dataset.index)]
    return train_dataset, test_dataset


if __name__ == '__main__':
    train_dataset, test_dataset = load_insurance_data('./insurance.csv')
    mlrm = LRModel(use_gradient_descent=False)
    mlrm.fit(train_dataset)
    mlrm.predict(test_dataset)
    # sk_linear_regression = LinearRegression()
    # sk_linear_regression.fit(train_dataset[train_dataset.columns[:-1]], train_dataset[train_dataset.columns[-1]])
    # y_ = sk_linear_regression.predict(test_dataset[test_dataset.columns[:-1]])
    # y = test_dataset[test_dataset.columns[-1]]
    # y_mean = np.mean(y)
    # n = test_dataset.shape[0]
    # rmse = np.sqrt(np.sum((y - y_) ** 2) / n)
    # r_square = 1 - np.sum((y - y_) ** 2) / np.sum((y - y_mean) ** 2)
    # print(rmse, r_square)
