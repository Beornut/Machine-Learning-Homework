import math

import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


class GaussianNaiveBayes:
    def __init__(self):
        self.prob_dict = {}

    def fit(self, dataset):
        classes = dataset[dataset.columns[-1]].value_counts(normalize=True)
        for i in classes.index:
            subset = dataset[dataset[dataset.columns[-1]] == i]
            self.prob_dict[i] = {}
            self.prob_dict[i]['prob'] = classes[i]
            for j in dataset.columns[:-1]:
                self.prob_dict[i][j] = {}
                self.prob_dict[i][j]['mean'] = subset[j].mean()
                self.prob_dict[i][j]['variance'] = subset[j].var()

    def predict(self, dataset):
        y_ = []
        y_prob = []
        feature = dataset.columns[:-1]
        for _, i in dataset.iterrows():
            probs = []
            max_prob = 0
            max_class = None
            for j in self.prob_dict.keys():
                prob = self.prob_dict[j]['prob']
                for k in feature:
                    mu = self.prob_dict[j][k]['mean']
                    sigma = self.prob_dict[j][k]['variance']
                    prob *= 1 / math.sqrt(2 * math.pi * sigma ** 2) * math.exp(-(i[k] - mu) / (2 * sigma ** 2))
                probs.append(prob)
                if prob > max_prob:
                    max_prob = prob
                    max_class = j
            prob_sum = sum(probs)
            probs = [p / prob_sum for p in probs]
            y_.append(max_class)
            y_prob.append(probs)
        return y_, y_prob


class MultinomialNaiveBayes:
    def __init__(self, laplace=1):
        self.laplace = laplace
        self.prob_dict = {}

    def fit(self, dataset):
        classes = dataset[dataset.columns[-1]].value_counts(normalize=True)
        n = len(dataset.columns[-1])
        for i in classes.index:
            subset = dataset[dataset[dataset.columns[-1]] == i]
            self.prob_dict[i] = {}
            self.prob_dict[i]['prob'] = classes[i]
            for j in dataset.columns[:-1]:
                self.prob_dict[i][j] = (subset[j].sum() + self.laplace) / (subset.sum().sum() + self.laplace * n)

    def predict(self, dataset):
        y_ = []
        y_prob = []
        feature = dataset.columns[:-1]
        for _, i in dataset.iterrows():
            probs = []
            max_prob = 0
            max_class = None
            for j in self.prob_dict.keys():
                prob = self.prob_dict[j]['prob']
                for k in feature:
                    prob *= math.pow(self.prob_dict[j][k], i[k])
                probs.append(prob)
                if prob > max_prob:
                    max_prob = prob
                    max_class = j
            prob_sum = sum(probs)
            probs = [p / prob_sum for p in probs]
            y_.append(max_class)
            y_prob.append(probs)
        return y_, y_prob


class BernoulliNaiveBayes:
    def __init__(self):
        self.prob_dict = {}

    def fit(self, dataset):
        classes = dataset[dataset.columns[-1]].value_counts(normalize=True)
        n = len(dataset.columns[-1])
        for i in classes.index:
            subset = dataset[dataset[dataset.columns[-1]] == i]
            self.prob_dict[i] = {}
            self.prob_dict[i]['prob'] = classes[i]
            for j in dataset.columns[:-1]:
                self.prob_dict[i][j] = subset[j].sum() / dataset.shape[0]

    def predict(self, dataset):
        y_ = []
        y_prob = []
        feature = dataset.columns[:-1]
        for _, i in dataset.iterrows():
            probs = []
            max_prob = 0
            max_class = None
            for j in self.prob_dict.keys():
                prob = self.prob_dict[j]['prob']
                for k in feature:
                    if i[k] == 1:
                        prob *= self.prob_dict[j][k]
                    else:
                        prob *= 1 - self.prob_dict[j][k]
                probs.append(prob)
                if prob > max_prob:
                    max_prob = prob
                    max_class = j
            prob_sum = sum(probs)
            probs = [p / prob_sum for p in probs]
            y_.append(max_class)
            y_prob.append(probs)
        return y_, y_prob


def load_diabetes_data_for_gaussian(path):
    all_data = pd.read_csv(path)
    train_dataset = all_data.sample(frac=0.7, random_state=0, axis=0)
    test_dataset = all_data[~all_data.index.isin(train_dataset.index)]
    return train_dataset, test_dataset


def load_diabetes_data_for_multinomial(path):
    all_data = pd.read_csv(path)
    all_data.drop('BMI', axis=1, inplace=True)
    all_data.drop('DiabetesPedigreeFunction', axis=1, inplace=True)
    train_dataset = all_data.sample(frac=0.7, random_state=0, axis=0)
    test_dataset = all_data[~all_data.index.isin(train_dataset.index)]
    return train_dataset, test_dataset


def load_bank_data_for_bernoulli(path):
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
    bank = pd.get_dummies(data=bank, columns=['job', 'marital', 'education', 'poutcome'],
                          prefix=['job', 'marital', 'education', 'poutcome'])
    label = bank['deposit']
    bank.drop('deposit', axis=1, inplace=True)
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


def test_gaussian():
    train_dataset, test_dataset = load_diabetes_data_for_gaussian('./diabetes.csv')
    gnb = GaussianNaiveBayes()
    gnb.fit(train_dataset)
    y_, y_prob = gnb.predict(test_dataset)
    sk_gnb = GaussianNB()
    sk_gnb.fit(train_dataset[train_dataset.columns[:-1]], train_dataset[train_dataset.columns[-1]])
    sk_y_ = sk_gnb.predict(test_dataset[test_dataset.columns[:-1]])
    y = test_dataset[test_dataset.columns[-1]]
    tp = fp = tn = fn = sk_tp = sk_fp = sk_tn = sk_fn = 0
    y_iter = iter(y_)
    sk_y_iter = iter(sk_y_)
    for i in y:
        if i == 1:
            if next(y_iter) == 1:
                tp += 1
            else:
                fn += 1
            if next(sk_y_iter) == 1:
                sk_tp += 1
            else:
                sk_fn += 1
        else:
            if next(y_iter) == 1:
                fp += 1
            else:
                tn += 1
            if next(sk_y_iter) == 1:
                sk_fp += 1
            else:
                sk_tn += 1
    print(pd.DataFrame([[tp, fp, tn, fn], [sk_tp, sk_fp, sk_tn, sk_fn]], index=['my_gaussian', 'sk_gaussian'],
                       columns=['tp', 'fp', 'tn', 'fn']))


def test_multinomial():
    train_dataset, test_dataset = load_diabetes_data_for_gaussian('./diabetes.csv')
    mnb = MultinomialNaiveBayes()
    mnb.fit(train_dataset)
    y_, y_prob = mnb.predict(test_dataset)
    sk_mnb = MultinomialNB()
    sk_mnb.fit(train_dataset[train_dataset.columns[:-1]], train_dataset[train_dataset.columns[-1]])
    sk_y_ = sk_mnb.predict(test_dataset[test_dataset.columns[:-1]])
    y = test_dataset[test_dataset.columns[-1]]
    tp = fp = tn = fn = sk_tp = sk_fp = sk_tn = sk_fn = 0
    y_iter = iter(y_)
    sk_y_iter = iter(sk_y_)
    for i in y:
        if i == 1:
            if next(y_iter) == 1:
                tp += 1
            else:
                fn += 1
            if next(sk_y_iter) == 1:
                sk_tp += 1
            else:
                sk_fn += 1
        else:
            if next(y_iter) == 1:
                fp += 1
            else:
                tn += 1
            if next(sk_y_iter) == 1:
                sk_fp += 1
            else:
                sk_tn += 1
    print(pd.DataFrame([[tp, fp, tn, fn], [sk_tp, sk_fp, sk_tn, sk_fn]], index=['my_multinomial', 'sk_multinomial'],
                       columns=['tp', 'fp', 'tn', 'fn']))


def test_bernoulli():
    train_dataset, test_dataset = load_bank_data_for_bernoulli('./bank.csv')
    bnb = BernoulliNaiveBayes()
    bnb.fit(train_dataset)
    y_, y_prob = bnb.predict(test_dataset)
    sk_bnb = BernoulliNB()
    sk_bnb.fit(train_dataset[train_dataset.columns[:-1]], train_dataset[train_dataset.columns[-1]])
    sk_y_ = sk_bnb.predict(test_dataset[test_dataset.columns[:-1]])
    y = test_dataset[test_dataset.columns[-1]]
    tp = fp = tn = fn = sk_tp = sk_fp = sk_tn = sk_fn = 0
    y_iter = iter(y_)
    sk_y_iter = iter(sk_y_)
    for i in y:
        if i == 1:
            if next(y_iter) == 1:
                tp += 1
            else:
                fn += 1
            if next(sk_y_iter) == 1:
                sk_tp += 1
            else:
                sk_fn += 1
        else:
            if next(y_iter) == 1:
                fp += 1
            else:
                tn += 1
            if next(sk_y_iter) == 1:
                sk_fp += 1
            else:
                sk_tn += 1
    print(pd.DataFrame([[tp, fp, tn, fn], [sk_tp, sk_fp, sk_tn, sk_fn]], index=['my_bernoulli', 'sk_bernoulli'],
                       columns=['tp', 'fp', 'tn', 'fn']))


if __name__ == '__main__':
    # print(load_diabetes_data_for_bernoulli('./bank.csv')[0].columns)
    test_gaussian()


