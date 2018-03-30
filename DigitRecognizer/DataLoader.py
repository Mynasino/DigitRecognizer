import pandas as pd 
import numpy as np 
import random

class DataLoad(object):
    def __init__(self):
        train_df = pd.read_csv('Data/train.csv')
        self.all_Y = np.array(train_df['label'])
        
        train_df.drop(['label'], axis=1, inplace=True)
        self.all_X = np.array(train_df).reshape((42000, 28, 28, 1))

    def data_iter(self, batch_size, n_train):
        inds = np.arange(n_train)
        random.shuffle(inds)

        n_batch = int(n_train / batch_size)
        if n_train % batch_size:
            n_batch = n_batch + 1

        for i in range(n_batch):
            batch_inds = inds[i * batch_size : min((i + 1) * batch_size, n_train)]
            yield self.all_X[batch_inds], self.all_Y[batch_inds]