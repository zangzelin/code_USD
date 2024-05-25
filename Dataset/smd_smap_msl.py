# -*- coding: utf-8 -*-
import os
import pickle
import torch
import numpy as np
from pynndescent import NNDescent

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd

prefix = "Data/input/processed"


def save_z(z, filename='z'):
    """
    save the sampled z in a txt file
    """
    for i in range(0, z.shape[1], 20):
        with open(filename + '_' + str(i) + '.txt', 'w') as file:
            for j in range(0, z.shape[0]):
                for k in range(0, z.shape[2]):
                    file.write('%f ' % (z[j][i][k]))
                file.write('\n')
    i = z.shape[1] - 1
    with open(filename + '_' + str(i) + '.txt', 'w') as file:
        for j in range(0, z.shape[0]):
            for k in range(0, z.shape[2]):
                file.write('%f ' % (z[j][i][k]))
            file.write('\n')


def get_data_dim(dataset):
    if dataset == 'SMAP':
        return 25
    elif dataset == 'MSL':
        return 55
    elif str(dataset).startswith('machine'):
        return 38
    else:
        raise ValueError('unknown dataset ' + str(dataset))


def load_smd_smap_msl(dataset, batch_size=512, window_size=60, stride_size=10, train_split=0.6, label=False,
                      do_preprocess=True, train_start=0,
                      test_start=0, k=10, alpha=0.5, seed=15):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """

    x_dim = get_data_dim(dataset)

    try:
        f = open(os.path.join(prefix, dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    print('testset size', test_label.shape, 'anomaly ration', sum(test_label) / len(test_label))

    whole_data = test_data
    whole_label = test_label
    print('testset size', whole_label.shape, 'anomaly ration', sum(whole_label) / len(whole_label))
    if do_preprocess:
        whole_data = preprocess(whole_data)

    n_sensor = whole_data.shape[1]
    print('n_sensor', n_sensor)

    train_df = whole_data[:int(train_split * len(whole_data))]
    train_label = whole_label[:int(train_split * len(whole_data))]

    # val_df = whole_data[int(0.6 * len(whole_data)):int(0.8 * len(whole_data))]
    # val_label = whole_label[int(0.6 * len(whole_data)):int(0.8 * len(whole_data))]

    test_df = whole_data[int(train_split * len(whole_data)):]
    test_label = whole_label[int(train_split * len(whole_data)):]

    print('train size', train_label.shape, 'anomaly ration', sum(train_label) / len(train_label))
    print('test size', test_label.shape, 'anomaly ration', sum(test_label) / len(test_label))

    if label:
        train_loader = DataLoader(
            Smd_smap_msl_dataset(train_df, train_label, window_size, stride_size, train_split=train_split,
                                 train='train', dataset=dataset),
            batch_size=batch_size, shuffle=False
        )
        # train_dataset = Smd_smap_msl_dataset(train_df, train_label, window_size, stride_size, train_split=train_split, train='train', dataset=dataset)
    else:
        train_loader = DataLoader(
            Smd_smap_msl_dataset(train_df, train_label, window_size, stride_size, train_split=train_split,
                                 train='train', k=k, alpha=alpha, dataset=dataset, seed=seed),
            batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False, persistent_workers=True
        )
        # train_dataset = Smd_smap_msl_dataset(train_df, train_label, window_size, stride_size, train_split=train_split, train='train', k=k, alpha=alpha, dataset=dataset)

    # val_loader = DataLoader(
    #     Smd_smap_msl_dataset(val_df, val_label, window_size, stride_size, train_split=train_split, train='val', k=k, alpha=alpha, dataset=dataset),
    #     batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
    # )

    test_loader = DataLoader(
        Smd_smap_msl_dataset(test_df, test_label, window_size, stride_size, train_split=train_split, train='test', k=k,
                             alpha=alpha, dataset=dataset, seed=seed),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False, persistent_workers=True
    )
    # test_dataset = Smd_smap_msl_dataset(test_df, test_label, window_size, stride_size, train_split=train_split, train='test', k=k, alpha=alpha, dataset=dataset)

    return train_loader, test_loader, n_sensor


def load_smd_smap_msl_occ(dataset, batch_size=512, window_size=60, stride_size=10, train_split=0.6, label=False,
                          do_preprocess=True, train_start=0,
                          test_start=0):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """

    x_dim = get_data_dim(dataset)
    f = open(os.path.join(prefix, dataset + '_train.pkl'), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:, :]

    f.close()
    try:
        f = open(os.path.join(prefix, dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None

    if do_preprocess:
        train_data = preprocess(train_data)
        test_data = preprocess(test_data)

    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", test_label.shape)
    n_sensor = train_data.shape[1]
    print('n_sensor', n_sensor)

    train_df = train_data[:]
    train_label = [0] * len(train_df)

    val_df = train_data[int(train_split * len(train_data)):]
    val_label = [0] * len(val_df)

    test_df = test_data[int(train_split * len(test_data)):]
    test_label = test_label[int(train_split * len(test_data)):]
    print('testset size', test_label.shape, 'anomaly ration', sum(test_label) / len(test_label))

    if label:
        train_loader = DataLoader(Smd_smap_msl_dataset(train_df, train_label, window_size, stride_size),
                                  batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(Smd_smap_msl_dataset(train_df, train_label, window_size, stride_size),
                                  batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Smd_smap_msl_dataset(val_df, val_label, window_size, stride_size), batch_size=batch_size,
                            shuffle=False)
    test_loader = DataLoader(Smd_smap_msl_dataset(test_df, test_label, window_size, stride_size), batch_size=batch_size,
                             shuffle=False)
    return train_loader, val_loader, test_loader, n_sensor


def preprocess(df, mode='Normal'):
    """returns normalized and standardized data.
    """

    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num()

    # normalize data
    if mode == 'Normal':
        df = StandardScaler().fit_transform(df)
    else:
        df = MinMaxScaler().fit_transform(df)
    print('Data normalized')

    return df



class Smd_smap_msl_dataset(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10, k=10, alpha=0.5, train_split=0.8, train='train',
                 dataset='MSL', seed=15) -> None:
        super(Smd_smap_msl_dataset, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size
        self.train_split = train_split

        self.data, self.idx, self.label = self.preprocess(df, label)
        # self.columns = np.append(df.columns, ["Label"])
        # self.timeindex = df.index[self.idx]
        print('label', self.label.shape, sum(self.label) / len(self.label))
        print('idx', self.idx.shape)
        print('data', self.data.shape)
        # print(len(self.data), len(self.idx), len(self.label))
        self.k = k
        self.alpha = alpha
        self.train = train
        self.dataset = dataset
        self.seed = seed
        filename = "save_near_index/{}_near{}_train{}_windowsize{}_trainsplit{}_seed{}.pkl".format(self.dataset, k,
                                                                                                   self.train,
                                                                                                   self.window_size,
                                                                                                   self.train_split,
                                                                                                   self.seed)
        if os.path.exists(filename):
            print('load near index from', filename)
            neighbors_index = pickle.load(open(filename, 'rb'))
        else:
            X_rshaped = self.data
            np.random.seed(self.seed)
            index = NNDescent(X_rshaped, n_jobs=-1)
            neighbors_index, neighbors_dist = index.query(X_rshaped, k=k + 1)
            neighbors_index = neighbors_index[:, 1:]
            pickle.dump(neighbors_index, open(filename, 'wb'))
            print('save near index to', filename)
        self.neighbors_index = neighbors_index

    def preprocess(self, df, label):
        start_idx = np.arange(0, len(df) - self.window_size, self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)

        label = [0 if sum(label[index:index + self.window_size]) == 0 else 1 for index in start_idx]
        return df, start_idx, np.array(label)

    def __len__(self):
        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 

        start = self.idx[index]
        end = start + self.window_size
        data_origin = self.data[start:end].reshape([self.window_size, -1, 1])

        augment_index = self.neighbors_index[start:end]

        # import pdb; pdb.set_trace()
        random_index = augment_index[:, np.random.choice(range(self.k), 1)]

        # data_origin = torch.tensor(self.data[index])
        data_near = self.data[random_index].reshape([self.window_size, -1, 1])
        alpha = np.random.uniform(0, self.alpha)
        augment_data = (alpha) * data_origin + (1 - alpha) * data_near

        return torch.FloatTensor(data_origin).transpose(0, 1), torch.FloatTensor(augment_data).transpose(0, 1), \
               self.label[index], index
