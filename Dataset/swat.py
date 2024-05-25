import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from pynndescent import NNDescent
import os
import pickle


class Augmenter():
    def __init__(self, train_df, k=10, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.data = train_df.to_numpy()

        os.mkdir('save_near_index') if not os.path.exists('save_near_index') else None
        filename = "save_near_index/SWat_near{}.pkl".format(k)
        if os.path.exists(filename):
            neighbors_index = pickle.load(open(filename, 'rb'))
        else:
            X_rshaped = train_df.values.reshape([train_df.shape[0], -1])
            index = NNDescent(X_rshaped, n_jobs=-1)
            neighbors_index, neighbors_dist = index.query(X_rshaped, k=k + 1)
            neighbors_index = neighbors_index[:, 1:]
            pickle.dump(neighbors_index, open(filename, 'wb'))
        # import pdb; pdb.set_trace()
        self.neighbors_index = neighbors_index

    def get_augment_data(self, index, device='cpu'):
        augment_index = self.neighbors_index[index]

        # import pdb; pdb.set_trace()
        random_index = augment_index[:, np.random.choice(range(self.k), 1)]

        data_origin = torch.tensor(self.data[index]).to(device)
        data_near = torch.tensor(self.data[random_index]).to(device)
        alpha = np.random.uniform(0, self.alpha)
        augment_data = (alpha) * data_origin + (1 - alpha) * data_near
        return augment_data


def loader_SWat(batch_size, window_size, stride_size, train_split, label=False, k=10, alpha=0.5, seed=15):
    root = "Data/input/SWaT_Dataset_Attack_v0.csv"
    data = pd.read_csv(root, sep=';', low_memory=False)
    Timestamp = pd.to_datetime(data["Timestamp"], format="mixed")
    data["Timestamp"] = Timestamp
    data = data.set_index("Timestamp")
    labels = [int(l != 'Normal') for l in data["Normal/Attack"].values]
    for i in list(data):
        data[i] = data[i].apply(lambda x: str(x).replace(",", "."))
    data = data.drop(["Normal/Attack"], axis=1)
    data = data.astype(float)
    n_sensor = len(data.columns)
    # %%
    feature = data.iloc[:, :51]
    scaler = StandardScaler()
    norm_feature = scaler.fit_transform(feature)

    norm_feature = pd.DataFrame(norm_feature, columns=data.columns, index=Timestamp)
    norm_feature = norm_feature.dropna(axis=1)

    train_df = norm_feature.iloc[:int(train_split * len(data))]
    train_label = labels[:int(train_split * len(data))]
    print('trainset size', train_df.shape, 'anomaly ration', sum(train_label) / len(train_label))

    # val_df = norm_feature.iloc[int(0.6 * len(data)):int(train_split * len(data))]
    # val_label = labels[int(0.6 * len(data)):int(train_split * len(data))]

    # test_df = norm_feature.iloc[int(train_split * len(data)):]
    # test_label = labels[int(train_split * len(data)):]

    test_df = norm_feature.iloc[int(0.8*len(data)):]
    test_label = labels[int(0.8*len(data)):]

    # augmenter.get_augment_data([0,2,3,6])

    print('testset size', test_df.shape, 'anomaly ration', sum(test_label) / len(test_label))
    if label:
        train_loader = DataLoader(
            SWat_dataset(train_df, train_label, window_size, stride_size, train='train'),
            batch_size=batch_size,
            shuffle=False,
        )
        # train_dataset = SWat_dataset(train_df, train_label, window_size, stride_size, train='train'),

    else:
        train_loader = DataLoader(
            SWat_dataset(train_df, train_label, window_size, stride_size, train='train', k=k, alpha=alpha,
                         train_split=train_split, seed=seed),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=False,
            persistent_workers=True,

        )
        # train_dataset = SWat_dataset(train_df, train_label, window_size, stride_size, train='train', k=k, alpha=alpha, train_split=train_split)

    # val_loader = DataLoader(
    #     SWat_dataset(val_df, val_label, window_size, stride_size, train='val', k=k, alpha=alpha, train_split=train_split),
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True,
    #     persistent_workers=True,
    # )

    test_loader = DataLoader(
        SWat_dataset(test_df, test_label, window_size, stride_size, train='test', k=k, alpha=alpha,
                     train_split=train_split, seed=seed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True,
    )

    # test_dataset = SWat_dataset(test_df, test_label, window_size, stride_size, train='test', k=k, alpha=alpha, train_split=train_split)

    return train_loader, test_loader, n_sensor


def loader_SWat_OCC(root, batch_size, window_size, stride_size, train_split, label=False):
    data = pd.read_csv("Data/input/SWaT_Dataset_Normal_v1.csv", sep=',', low_memory=False)
    Timestamp = pd.to_datetime(data["Timestamp"])
    data["Timestamp"] = Timestamp
    data = data.set_index("Timestamp")
    labels = [int(l != 'Normal') for l in data["Normal/Attack"].values]
    for i in list(data):
        data[i] = data[i].apply(lambda x: str(x).replace(",", "."))
    data = data.drop(["Normal/Attack"], axis=1)
    data = data.astype(float)
    n_sensor = len(data.columns)
    # %%
    feature = data.iloc[:, :51]
    scaler = StandardScaler()

    norm_feature = scaler.fit_transform(feature)
    norm_feature = pd.DataFrame(norm_feature, columns=data.columns, index=Timestamp)
    norm_feature = norm_feature.dropna(axis=1)
    train_df = norm_feature.iloc[:]
    train_label = labels[:]
    print('trainset size', train_df.shape, 'anomaly ration', sum(train_label) / len(train_label))

    val_df = norm_feature.iloc[int(train_split * len(data)):]
    val_label = labels[int(train_split * len(data)):]

    data = pd.read_csv('Data/input/SWaT_Dataset_Attack_v0.csv', sep=';', low_memory=False)
    Timestamp = pd.to_datetime(data["Timestamp"])
    data["Timestamp"] = Timestamp
    data = data.set_index("Timestamp")
    labels = [int(l != 'Normal') for l in data["Normal/Attack"].values]
    for i in list(data):
        data[i] = data[i].apply(lambda x: str(x).replace(",", "."))
    data = data.drop(["Normal/Attack"], axis=1)
    data = data.astype(float)
    n_sensor = len(data.columns)

    feature = data.iloc[:, :51]
    scaler = StandardScaler()
    norm_feature = scaler.fit_transform(feature)
    norm_feature = pd.DataFrame(norm_feature, columns=data.columns, index=Timestamp)
    norm_feature = norm_feature.dropna(axis=1)

    test_df = norm_feature.iloc[int(0.8 * len(data)):]
    test_label = labels[int(0.8 * len(data)):]

    print('testset size', test_df.shape, 'anomaly ration', sum(test_label) / len(test_label))
    if label:
        train_loader = DataLoader(SWat_dataset(train_df, train_label, window_size, stride_size, train='train'),
                                  batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(SWat_dataset(train_df, train_label, window_size, stride_size, train='train'),
                                  batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SWat_dataset(val_df, val_label, window_size, stride_size, train='val'),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SWat_dataset(test_df, test_label, window_size, stride_size, train='test'),
                             batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, n_sensor


class SWat_dataset(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10, k=10, alpha=0.5, train_split=0.8,
                 train='train', seed=15) -> None:
        super(SWat_dataset, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size
        self.train_split = train_split

        self.data, self.idx, self.label = self.preprocess(df, label)
        self.columns = np.append(df.columns, ["Label"])
        self.timeindex = df.index[self.idx]
        self.k = k
        self.alpha = alpha
        self.train = train
        self.seed = seed
        filename = "save_near_index/SWat_near{}_train{}_windowsize{}_trainsplit{}_seed{}_622.pkl".format(k, self.train,
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

        delat_time = df.index[end_idx] - df.index[start_idx]

        idx_mask = delat_time == pd.Timedelta(self.window_size, unit='s')

        start_index = start_idx[idx_mask]

        label = [0 if sum(label[index:index + self.window_size]) == 0 else 1 for index in start_index]
        return df.values, start_idx[idx_mask], np.array(label)

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D
        # output:(51,60,1)
        """
        """
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
               self.label[index], start
