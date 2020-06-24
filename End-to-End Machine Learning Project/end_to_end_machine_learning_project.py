#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split


class Housing(object):

    def __init__(self):
        self.data_path = '/Users/mashuan/PycharmProjects/machine_learning_algorithm_practice/housing.csv'

    def load_housing_data(self):
        return pd.read_csv(self.data_path)

    def split_train_test(self, data, test_ratio):
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return data.iloc[train_indices], data.iloc[test_indices]

    def test_set_check(self, identifier, test_ratio, hash):
        return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

    def split_train_test_by_id(self, data, test_ratio, id_column, hash=hashlib.md5):
        ids = data[id_column]
        in_test_set = ids.apply(lambda id_: self.test_set_check(id_, test_ratio, hash))
        return data.loc[~in_test_set], data.loc[in_test_set]


if __name__ == '__main__':
    housing = Housing()
    # load data
    housing_data = housing.load_housing_data()
    # show top 5 records
    print housing_data.head()
    # describe data
    print housing_data.info()
    # count statistics
    print housing_data['ocean_proximity'].value_counts()
    # statistics info
    print housing_data.describe()
    # show distribution info
    housing_data.hist(bins=50, figsize=(20, 15))
    plt.show()
    # split data
    test_set_ratio = 0.2
    train_set, test_set = housing.split_train_test(housing_data, test_set_ratio)
    print len(train_set), 'train +', len(test_set), 'test'

    # add index as a new column 'index'
    housing_with_id = housing_data.reset_index()
    train_set, test_set = housing.split_train_test_by_id(housing_with_id, test_set_ratio, 'index')

    # use lng-lat as unique column 'id'
    # housing_with_id['id'] = housing_data['longitude'] * 1000 + housing_data['latitude']
    # train_set, test_set = housing.split_train_test_by_id(housing_with_id, test_set_ratio, 'id')

    # use sk-learn
    # train_set, test_set = train_test_split(housing, test_size=test_set_ratio, random_state=42)
