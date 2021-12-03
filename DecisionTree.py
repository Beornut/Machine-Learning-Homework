import os
import math
import pickle
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree


class DecisionTree:
    def __init__(self, max_depth=5, use_entropy=True):
        self.max_depth = max_depth
        self.use_entropy = use_entropy
        self.root = {}

    def __calc_entropy(self, dataset):
        res = 0
        for i in dataset[dataset.columns[-1]].value_counts(normalize=True).values:
            res -= i * math.log2(i)
        return res

    def __calc_gini(self, dataset):
        res = 1
        for i in dataset[dataset.columns[-1]].value_counts(normalize=True).values:
            res -= i ** 2
        return res

    def __build_tree(self, dataset, depth):
        node = {}
        if dataset.columns.size < 2 or dataset[dataset.columns[-1]].value_counts().size < 2 or depth == self.max_depth:
            count = dataset[dataset.columns[-1]].value_counts()
            return count.index[0]
        else:
            split_ig = 0
            split_feature = None
            if self.use_entropy:
                current_entropy = self.__calc_entropy(dataset)
                for i in dataset.columns[:-1]:
                    split_entropy = 0
                    count = dataset[i].value_counts(normalize=True)
                    for j in count.index:
                        split_entropy += count[j] * self.__calc_entropy(dataset[dataset[i] == j])
                    ig = current_entropy - split_entropy
                    if ig > split_ig:
                        split_ig = ig
                        split_feature = i
                if split_ig == 0:
                    count = dataset[dataset.columns[-1]].value_counts()
                    return count.index[0]
                node[split_feature] = {}
                for i in dataset[split_feature].value_counts().index:
                    node[split_feature][i] = self.__build_tree(
                        dataset[dataset[split_feature] == i].drop(split_feature, axis=1, inplace=False),
                        depth + 1)
            else:
                split_value = None
                current_gini = self.__calc_gini(dataset)
                for i in dataset.columns[:-1]:
                    count = dataset[i].value_counts(normalize=True)
                    for j in count.index:
                        split_gini = 0
                        left = dataset[dataset[i] <= j]
                        right = dataset[dataset[i] > j]
                        if left.shape[0] > 0 and right.shape[0] > 0:
                            split_gini += left.shape[0] / dataset.shape[0] * self.__calc_gini(left)
                            split_gini += right.shape[0] / dataset.shape[0] * self.__calc_gini(right)
                            ig = current_gini - split_gini
                            if ig > split_ig:
                                split_ig = ig
                                split_feature = i
                                split_value = j
                if split_ig == 0:
                    count = dataset[dataset.columns[-1]].value_counts()
                    return count.index[0]
                node[split_feature] = {}
                node[split_feature]['split_value'] = split_value
                node[split_feature]['left'] = self.__build_tree(
                    dataset[dataset[split_feature] <= split_value].drop(split_feature, axis=1, inplace=False),
                    depth + 1)
                node[split_feature]['right'] = self.__build_tree(
                    dataset[dataset[split_feature] > split_value].drop(split_feature, axis=1, inplace=False), depth + 1)
            return node

    def fit(self, dataset):
        self.root = self.__build_tree(dataset, 0)

    def test(self, dataset):
        tp = tn = fp = fn = 0
        for k, v in dataset.iterrows():
            if self.use_entropy:
                current_node = self.root
                while type(current_node) == dict:
                    current_feature = next(iter(current_node))
                    current_node = current_node[current_feature][v[current_feature]]
                if current_node == 1:
                    if v[-1] == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if v[-1] == 1:
                        fn += 1
                    else:
                        tn += 1
            else:
                current_node = self.root
                while type(current_node) == dict:
                    current_feature = next(iter(current_node))
                    current_val = current_node[current_feature]['split_value']
                    if v[current_feature] <= current_val:
                        current_node = current_node[current_feature]['left']
                    else:
                        current_node = current_node[current_feature]['right']
                if current_node == 1:
                    if v[-1] == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if v[-1] == 1:
                        fn += 1
                    else:
                        tn += 1
        return np.mat([
            [tp, fp],
            [fn, tn]
        ])


def load_bank_data(path, use_entropy=True):
    bank = pd.read_csv(path)
    bank['job'] = bank['job'].replace(['management', 'admin.', 'entrepreneur'], 'white-collar')
    bank['job'] = bank['job'].replace(['services', 'housemaid'], 'pink-collar')
    bank['job'] = bank['job'].replace(['technician'], 'blue-collar')
    bank['job'] = bank['job'].replace(['retired', 'student', 'unemployed', 'unknown', 'self-employed'], 'other')
    bank['poutcome'] = bank['poutcome'].replace(['other'], 'unknown')
    bank.drop('contact', axis=1, inplace=True)
    bank['default'] = bank['default'].map({'yes': 1, 'no': 0})
    bank['housing'] = bank['housing'].map({'yes': 1, 'no': 0})
    bank['loan'] = bank['loan'].map({'yes': 1, 'no': 0})
    bank.drop('month', axis=1, inplace=True)
    bank.drop('day', axis=1, inplace=True)
    bank["deposit"] = bank['deposit'].map({'yes': 1, 'no': 0})
    bank.loc[bank['pdays'] == -1, 'pdays'] = 10000
    bank['pdays'] = np.where(bank['pdays'], 1 / bank.pdays, 1 / bank.pdays)
    bank = pd.get_dummies(data=bank, columns=['job', 'marital', 'education', 'poutcome'],
                          prefix=['job', 'marital', 'education', 'poutcome'])
    label = bank['deposit']
    bank.drop('deposit', axis=1, inplace=True)
    if use_entropy:
        bank.drop('campaign', axis=1, inplace=True)
        bank.drop('age', axis=1, inplace=True)
        bank.drop('balance', axis=1, inplace=True)
        bank.drop('duration', axis=1, inplace=True)
        bank.drop('previous', axis=1, inplace=True)
        bank.drop('pdays', axis=1, inplace=True)
    bank.insert(bank.columns.size, 'deposit', label)
    train_dataset = bank.sample(frac=0.7, random_state=0, axis=0)
    test_dataset = bank[~bank.index.isin(train_dataset.index)]
    return train_dataset, test_dataset


def plot_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('Real')
    plt.ylabel('Predicted')
    plt.show()


if __name__ == '__main__':
    use_entropy = False
    if use_entropy:
        pkl_path = 'entropy_tree.pkl'
    else:
        pkl_path = 'gini_tree.pkl'
    train_dataset, test_dataset = load_bank_data('./bank.csv', use_entropy=use_entropy)
    # dt = DecisionTree(use_entropy=use_entropy)
    # if os.path.exists(pkl_path):
    #     with open(pkl_path, 'rb') as file:
    #         dt.root = pickle.load(file)
    # else:
    #     dt.fit(train_dataset)
    #     with open(pkl_path, 'wb') as file:
    #         pickle.dump(dt.root, file)
    # cm = dt.test(test_dataset)
    # plot_matrix(cm, ['deposit', 'not_deposit'])

    sk_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=5)
    sk_tree.fit(train_dataset[train_dataset.columns[:-1]], train_dataset[train_dataset.columns[-1]])
    feature_names = train_dataset.columns[:-1].to_list()
    sk_y = sk_tree.predict(test_dataset[test_dataset.columns[:-1]])
    tp = tn = fp = fn = 0
    iter_sk_y = iter(sk_y)
    for k, v in test_dataset.iterrows():
        if next(iter_sk_y) == 1:
            if v[-1] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if v[-1] == 1:
                fn += 1
            else:
                tn += 1
    print(tp, fn)
    # plot_matrix(np.mat([[tp, fp], [fn, tn]]), ['deposit', 'not_deposit'])
    tree.plot_tree(sk_tree, feature_names=feature_names)
    res = tree.plot_tree(sk_tree, feature_names=feature_names)
    plt.show()
