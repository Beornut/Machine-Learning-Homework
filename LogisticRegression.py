import numpy as np
import pandas as pd
import random

from sklearn import metrics
from sklearn.linear_model import LogisticRegression


class LRModel:
    def __init__(self, optimizer='SGD', batch_size=10, learning_rate=1e-3, penalty=None, penalty_coef=1e-1):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.penalty_coef = penalty_coef
        self.w = []

    def __gradient_decent(self, dataset):
        x = np.c_[np.zeros((dataset.shape[0], 1)), dataset[dataset.columns[:-1]]]
        y = dataset[dataset.columns[-1]]
        if len(self.w) < 1:
            self.w = np.zeros(x.shape[1])
        y_ = 1 / (1 + np.exp(-np.matmul(x, self.w)))
        n = x.shape[0]
        if self.penalty == 'l1':
            self.w = self.w - self.learning_rate * np.matmul(x.T, y_ - y) / n - self.penalty_coef * np.sign(self.w)
        elif self.penalty == 'l2':
            self.w = self.w - self.learning_rate * np.matmul(x.T, y_ - y) / n - self.penalty_coef * self.w
        else:
            self.w = self.w - self.learning_rate * np.matmul(x.T, y_ - y) / n

    def fit(self, dataset):
        i = 0
        if self.optimizer == 'SGD':
            while i + self.batch_size < dataset.shape[0]:
                idx = random.randint(i, i + self.batch_size - 1)
                samples = dataset[idx: idx + 1]
                self.__gradient_decent(samples)
                i += self.batch_size
            idx = random.randint(i, dataset.shape[0] - 1)
            samples = dataset[idx: idx + 1]
            self.__gradient_decent(samples)
        elif self.optimizer == 'BGD':
            self.__gradient_decent(dataset)
        elif self.optimizer == 'MBGD':
            while i + self.batch_size < dataset.shape[0]:
                samples = dataset[i: i + self.batch_size]
                self.__gradient_decent(samples)
                i += self.batch_size
            samples = dataset[i:]
            self.__gradient_decent(samples)

    def predict(self, dataset):
        x = np.c_[np.zeros((dataset.shape[0], 1)), dataset[dataset.columns[:-1]]]
        return 1 / (1 + np.exp(-np.matmul(x, self.w)))


def load_diabetes_data(path):
    return pd.read_csv(path)


if __name__ == '__main__':
    all_data = load_diabetes_data('./diabetes.csv')
    k_fold = 5
    fold_size = all_data.shape[0] // k_fold
    for i in range(k_fold):
        test_dataset = all_data[i * fold_size: i * fold_size + fold_size]
        train_dataset = all_data[~all_data.index.isin(test_dataset.index)]
        lrm = LRModel(optimizer='BGD', penalty='l2')
        lrm.fit(train_dataset)
        y = test_dataset[test_dataset.columns[-1]]
        y_ = lrm.predict(test_dataset)
        fpr, tpr, threholds = metrics.roc_curve(y, y_)
        auc = metrics.auc(fpr, tpr)
        sk_lrm = LogisticRegression(max_iter=1000)
        sk_lrm.fit(train_dataset[train_dataset.columns[:-1]], train_dataset[train_dataset.columns[-1]])
        sk_y_ = sk_lrm.predict_proba(test_dataset[test_dataset.columns[:-1]])
        sk_fpr, sk_tpr, sk_threholds = metrics.roc_curve(y, sk_y_[:, 1])
        sk_auc = metrics.auc(sk_fpr, sk_tpr)
        print(auc, sk_auc)
