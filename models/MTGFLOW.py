# %%
from cgitb import reset
from turtle import forward, shape
from numpy import percentile
import torch.nn as nn
import torch.nn.functional as F
from models.NF import MAF
import torch
import numpy as np
import scipy


def interpolate(tensor, index, target_size, mode='nearest', dim=0):
    print(tensor.shape)
    source_length = tensor.shape[dim]
    if source_length > target_size:
        raise AttributeError('no need to interpolate')
    if dim == -1:
        new_tensor = torch.zeros((*tensor.shape[:-1], target_size), dtype=tensor.dtype, device=tensor.device)
    if dim == 0:
        new_tensor = torch.zeros((target_size, *tensor.shape[1:],), dtype=tensor.dtype, device=tensor.device)
    scale = target_size // source_length
    reset = target_size % source_length
    # if mode == 'nearest':
    new_index = index
    new_tensor[new_index, :] = tensor
    new_tensor[:new_index[0], :] = tensor[0, :].unsqueeze(0)
    for i in range(source_length - 1):
        new_tensor[new_index[i]:new_index[i + 1], :] = tensor[i, :].unsqueeze(0)
    new_tensor[new_index[i + 1]:, :] = tensor[i + 1, :].unsqueeze(0)
    return new_tensor


class GNN(nn.Module):
    """
    The GNN module applied in GANF
    """

    def __init__(self, input_size, hidden_size):
        super(GNN, self).__init__()
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, A):
        ## A: K X K
        ## H: N X K  X L X D
        # print(h.shape, A.shape)
        # h_n = self.lin_n(torch.einsum('nkld,kj->njld',h,A))
        # h_n = self.lin_n(torch.einsum('nkld,kj->njld',h,A))
        # print(h.shape, A.shape)
        h_n = self.lin_n(torch.einsum('nkld,nkj->njld', h, A))
        h_r = self.lin_r(h[:, :, :-1])
        h_n[:, :, 1:] += h_r
        h = self.lin_2(F.relu(h_n))

        return h


import math
import torch.nn as nn
import matplotlib.pyplot as plt


def plot_attention(data, i, X_label=None, Y_label=None):
    '''
      Plot the attention model heatmap
      Args:
        data: attn_matrix with shape [ty, tx], cutted before 'PAD'
        X_label: list of size tx, encoder tags
        Y_label: list of size ty, decoder tags
    '''
    fig, ax = plt.subplots(figsize=(20, 8))  # set figure size
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.9)
    fig.colorbar(heatmap)
    # Set axis labels
    if X_label != None and Y_label != None:
        X_label = [x_label for x_label in X_label]
        Y_label = [y_label for y_label in Y_label]

        xticks = range(0, len(X_label))
        ax.set_xticks(xticks, minor=False)  # major ticks
        ax.set_xticklabels(X_label, minor=False, rotation=45)  # labels should be 'unicode'

        yticks = range(0, len(Y_label))
        ax.set_yticks(yticks, minor=False)
        ax.set_yticklabels(Y_label[::-1], minor=False)  # labels should be 'unicode'

        ax.grid(True)
        plt.show()
        plt.savefig('graph/attention{:04d}.jpg'.format(i))


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, c):
        super(ScaleDotProductAttention, self).__init__()
        self.w_q = nn.Linear(c, c)
        self.w_k = nn.Linear(c, c)
        self.w_v = nn.Linear(c, c)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)
        # swat_0.2

    def forward(self, x, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        shape = x.shape
        x_shape = x.reshape((shape[0], shape[1], -1))
        batch_size, length, c = x_shape.size()
        q = self.w_q(x_shape)
        k = self.w_k(x_shape)
        k_t = k.view(batch_size, c, length)  # transpose
        score = (q @ k_t) / math.sqrt(c)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        # 3. pass them softmax to make [0, 1] range
        score = self.dropout(self.softmax(score))

        return score, k


class MTGFLOW(nn.Module):

    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, window_size, n_sensor, dropout=0.1, model="MAF",
                 batch_norm=True):
        super(MTGFLOW, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, dropout=dropout)
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size)
        if model == "MAF":
            # self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm,activation='tanh', mode = 'zero')
            self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size,
                          batch_norm=batch_norm, activation='tanh')

        self.attention = ScaleDotProductAttention(window_size * input_size)

    def forward(self, x, ):
        return self.test(x, ).mean()

    def test(self, x, ):
        # x: N X K X L X D
        full_shape = x.shape
        graph, _ = self.attention(x)
        self.graph = graph
        # reshape: N*K, L, D
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        h, _ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))
        h = self.gcn(h, graph)

        # reshappe N*K*L,H
        h = h.reshape((-1, h.shape[3]))
        x = x.reshape((-1, full_shape[3]))
        log_prob = self.nf.log_prob(x, full_shape[1], full_shape[2], h).reshape([full_shape[0], -1])  #
        log_prob = log_prob.mean(dim=1)

        return log_prob

    def get_graph(self):
        return self.graph

    def locate(self, x, ):
        # x: N X K X L X D
        full_shape = x.shape

        graph, _ = self.attention(x)
        # reshape: N*K, L, D
        self.graph = graph
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        h, _ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))
        h = self.gcn(h, graph)

        # reshappe N*K*L,H
        h = h.reshape((-1, h.shape[3]))
        x = x.reshape((-1, full_shape[3]))
        a = self.nf.log_prob(x, full_shape[1], full_shape[2], h)
        log_prob, z = a[0].reshape([full_shape[0], full_shape[1], -1]), a[1].reshape([full_shape[0], full_shape[1], -1])

        return log_prob.mean(dim=2), z.reshape((full_shape[0] * full_shape[1], -1))


class MTGFLOWZL(nn.Module):

    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, window_size, n_sensor, dropout=0.1, model="MAF",
                 batch_norm=True):
        super(MTGFLOWZL, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, dropout=dropout)
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size)
        if model == "MAF":
            # self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm,activation='tanh', mode = 'zero')
            self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size,
                          batch_norm=batch_norm, activation='tanh')

        self.gcn_mlp = nn.Sequential(
            nn.Linear(n_sensor * window_size, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, n_sensor * window_size * 32),
        )

        self.ann = nn.Sequential(
            nn.Linear(n_sensor * window_size * 32, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
        )
        self.attention = ScaleDotProductAttention(window_size * input_size)

    def forward(self, x, ):
        hid, log_prob, h_gcn = self.test(x, )
        return hid, log_prob.mean(), h_gcn, log_prob

    def test(self, x, ):
        # x: N X K X L X D 
        # import pdb; pdb.set_trace()
        full_shape = x.shape
        graph, _ = self.attention(x)
        self.graph = graph
        # reshape: N*K, L, D
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        h, _ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))
        h_gcn = self.gcn(h, graph)
        # import pdb; pdb.set_trace()

        shape_hgcn = h_gcn.shape
        h_gcn = self.gcn_mlp(x.reshape((shape_hgcn[0], -1))).reshape(*shape_hgcn) + h_gcn
        h_gcn_for_ann = h_gcn.reshape(full_shape[0], -1)

        # reshappe N*K*L,H
        h_gcn_reshape = h_gcn.reshape((-1, h_gcn.shape[3]))
        x = x.reshape((-1, full_shape[3]))

        log_prob = self.nf.log_prob(x, full_shape[1], full_shape[2], h_gcn_reshape).reshape([full_shape[0], -1])  #
        log_prob = log_prob.mean(dim=1)
        return self.ann(h_gcn_for_ann), log_prob, h_gcn

    def get_graph(self):
        return self.graph

    def locate(self, x, ):
        # x: N X K X L X D 
        full_shape = x.shape

        graph, _ = self.attention(x)
        # reshape: N*K, L, D
        self.graph = graph
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        h, _ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))
        h = self.gcn(h, graph)

        # reshappe N*K*L,H
        h = h.reshape((-1, h.shape[3]))
        x = x.reshape((-1, full_shape[3]))
        a = self.nf.log_prob(x, full_shape[1], full_shape[2], h)
        import pdb;
        pdb.set_trace()

        log_prob, z = a[0].reshape([full_shape[0], full_shape[1], -1]), a[1].reshape([full_shape[0], full_shape[1], -1])

        return log_prob.mean(dim=2), z.reshape((full_shape[0] * full_shape[1], -1))

    def _DistanceSquared(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-6)

        return dist

    def _CalGamma(self, v):
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b

        return out

    def _TwowaydivergenceLoss(self, P_, Q_, select=None):
        EPS = 1e-5
        losssum1 = P_ * torch.log(Q_ + EPS)
        losssum2 = (1 - P_) * torch.log(1 - Q_ + EPS)
        losssum = -1 * (losssum1 + losssum2)
        return losssum.mean()

    def _Similarity(self, dist, gamma, v=100, h=1, pow=2):
        dist_rho = dist

        dist_rho[dist_rho < 0] = 0
        Pij = (
                gamma
                * torch.tensor(2 * 3.14)
                * gamma
                * torch.pow((1 + dist_rho / v), exponent=-1 * (v + 1))
        )
        return Pij

    def LossManifold(
            self,
            input_data,
            latent_data,
            v_input,
            v_latent,
            w=1,
            metric="euclidean",
            label=None,
    ):
        # normalize the input and latent data
        # import pdb; pdb.set_trace()
        # input_data = input_data / torch.std(input_data, dim=0).detach()
        # latent_data = latent_data / torch.std(latent_data, dim=0).detach()
        print("input_data:", input_data.shape)
        print("latent_data:", latent_data.shape)

        batch_size = input_data.shape[0]

        data_1 = input_data[: input_data.shape[0] // 2]
        dis_P = self._DistanceSquared(data_1, data_1)
        print("dis_P:", dis_P.shape)

        latent_data_1 = latent_data[: input_data.shape[0] // 2]

        dis_P_2 = dis_P  # + nndistance.reshape(1, -1)

        P_2 = self._Similarity(dist=dis_P_2, gamma=self._CalGamma(v_input), v=v_input)
        print("P_2:", P_2.shape)

        latent_data_2 = latent_data[(input_data.shape[0] // 2):]
        dis_Q_2 = self._DistanceSquared(latent_data_1, latent_data_2)
        print("dis_Q_2:", dis_Q_2.shape)

        Q_2 = self._Similarity(
            dist=dis_Q_2,
            gamma=self._CalGamma(v_latent),
            v=v_latent,
        )
        print("Q_2:", Q_2.shape)

        eye_mask = torch.eye(P_2.shape[0]).to(input_data.device)
        loss_ce_posi = self._TwowaydivergenceLoss(
            P_=P_2[eye_mask == 1], Q_=Q_2[eye_mask == 1]
        )
        # P_2_copy = P_2.detach()
        # label_matrix = label.reshape(1, -1).repeat(label.shape[0],1)
        # lable_mask = (label_matrix-label_matrix.T)!=0
        # P_2_copy[lable_mask] = 0.0
        loss_ce_nega = self._TwowaydivergenceLoss(
            P_=P_2[eye_mask == 0], Q_=Q_2[eye_mask == 0]
        )
        w1, w2 = 1 / (1 + w), w / (1 + w)
        return w2 * loss_ce_nega, w1 * loss_ce_posi / batch_size


class test(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, window_size, n_sensor, dropout=0.1, model="MAF",
                 batch_norm=True):
        super(test, self).__init__()

        if model == "MAF":
            self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, batch_norm=batch_norm,
                          activation='tanh', mode='zero')
        self.attention = ScaleDotProductAttention(window_size * input_size)

    def forward(self, x, ):
        return self.test(x, ).mean()

    def test(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        full_shape = x.shape
        x = x.reshape((full_shape[0] * full_shape[1], full_shape[2], full_shape[3]))
        x = x.reshape((-1, full_shape[3]))
        log_prob = self.nf.log_prob(x, full_shape[1], full_shape[2]).reshape(
            [full_shape[0], full_shape[1], -1])  # *full_shape[1]*full_shape[2]
        log_prob = log_prob.mean(dim=1)
        return log_prob

    def locate(self, x, ):
        # x: N X K X L X D 
        x = x.unsqueeze(2).unsqueeze(3)
        full_shape = x.shape

        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        # reshappe N*K*L,H
        x = x.reshape((-1, full_shape[3]))
        a = self.nf.log_prob(x, full_shape[1], full_shape[2])  # *full_shape[1]*full_shape[2]
        log_prob, z = a[0].reshape([full_shape[0], full_shape[1], -1]), a[1].reshape([full_shape[0], full_shape[1], -1])

        return log_prob.mean(dim=2), z.reshape((full_shape[0] * full_shape[1], -1))
