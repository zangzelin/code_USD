# %%
import json
import os
import argparse
import torch
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from models.MTGFLOW import MTGFLOW, MTGFLOWZL
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, f1_score
import pickle
import wandb

parser = argparse.ArgumentParser()

# parser.add_argument('--data_dir', type=str,
#                     default='Data/input/SWaT_Dataset_Attack_v0.csv', help='Location of datasets.')
parser.add_argument('--output_dir', type=str,
                    default='./checkpoint/')
parser.add_argument('--name', default='SWaT', help='the name of dataset')

parser.add_argument('--graph', type=str, default='None')
parser.add_argument('--model', type=str, default='MAF')

parser.add_argument('--n_blocks', type=int, default=1,
                    help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--n_components', type=int, default=1,
                    help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=32,
                    help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--input_size', type=int, default=1)
parser.add_argument('--batch_norm', type=bool, default=False)
parser.add_argument('--train_split', type=float, default=0.6)
parser.add_argument('--stride_size', type=int, default=10)

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--window_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')
parser.add_argument('--seed', type=int, default=15, help='Learning rate.')
parser.add_argument('--log_emb', type=bool, default=False)
parser.add_argument('--save_model', type=bool, default=False)
parser.add_argument('--log_loss', type=bool, default=False)

parser.add_argument('--v_latent', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--loss_weight_manifold_po', type=float, default=0.1)
parser.add_argument('--loss_weight_manifold_ne', type=float, default=0.1)
parser.add_argument('--epoch', type=int, default=400)

args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


import random
import numpy as np

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# %%
print("Loading dataset")
print(args.name)
from Dataset import load_smd_smap_msl, loader_SWat, loader_WADI, loader_PSM, loader_WADI_OCC

if args.name == 'SWaT':
    train_loader, test_loader, n_sensor = loader_SWat(
        args.batch_size, args.window_size, args.stride_size, args.train_split, k=args.k,
        alpha=args.alpha,
        seed=args.seed
    )

elif args.name == 'Wadi':
    train_loader, test_loader, n_sensor = loader_WADI(
        args.batch_size, args.window_size, args.stride_size, args.train_split, k=args.k,
        alpha=args.alpha,
        seed=args.seed
    )

elif args.name == 'SMAP' or args.name == 'MSL' or args.name.startswith('machine'):
    train_loader, test_loader, n_sensor = load_smd_smap_msl(
        args.name, args.batch_size, args.window_size, args.stride_size, args.train_split, k=args.k,
        alpha=args.alpha,
        seed=args.seed
    )

elif args.name == 'PSM':
    train_loader, test_loader, n_sensor = loader_PSM(
        args.name, args.batch_size, args.window_size, args.stride_size, args.train_split, k=args.k,
        alpha=args.alpha,
        seed=args.seed
    )

print("n_sensor:", n_sensor)

# %%
model = MTGFLOWZL(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.window_size,
                  n_sensor,
                  dropout=0.0, model=args.model, batch_norm=args.batch_norm)

model = model.to(device)

save_path = os.path.join(args.output_dir, args.name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# %%
from torch.nn.utils import clip_grad_value_

loss_best = 100
roc_max = 0
ap_max = 0
best_feature_data = {}

lr = args.lr
optimizer = torch.optim.AdamW([
    {'params': model.parameters(), 'weight_decay': args.weight_decay},
], lr=lr, weight_decay=args.weight_decay)

now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")

for epoch in range(args.epoch):
    test_input_data_list = []
    test_h_data_list = []
    test_h_label_list = []
    train_h_data_list = []
    train_h_label_list = []
    train_input_data_list = []
    train_log_prob_list = []
    train_idx_list = []
    test_idx_list = []
    test_log_prob_list = []

    loss_train = []
    model.train()
    for x_ori, x_aug, label, idx in train_loader:
        x_ori = x_ori.to(device)
        x_aug = x_aug.to(device)
        x_all = torch.cat([x_ori, x_aug], dim=0)

        optimizer.zero_grad()
        hid, loss, _, _ = model(x_all, ) 

        loss_mani_po, loss_mani_ne = model.LossManifold(
            input_data=x_all.reshape(x_all.shape[0], -1),
            latent_data=hid.reshape(x_all.shape[0], -1),
            v_input=100,
            v_latent=args.v_latent,
        )

        total_loss = -loss + loss_mani_po * args.loss_weight_manifold_po + loss_mani_ne * args.loss_weight_manifold_ne

        total_loss.backward()
        clip_grad_value_(model.parameters(), 1)
        optimizer.step()

    model.eval()
    loss_test = []
    device_test = torch.device("cpu")
    with torch.no_grad():
        for x, x_aug, label, idx in test_loader:
            x = x.to(device)
            h_vis, log_prob, h_gcn_test = model.test(x, ) 

            h_vis = h_vis.to(device_test)
            log_prob = log_prob.to(device_test)
            h_gcn_test = h_gcn_test.to(device_test)

            loss = -log_prob.cpu().numpy()
            loss_test.append(loss)

            if args.log_emb:
                # test_input_data_list.append(x.cpu().detach().numpy().reshape(x.shape[0], -1))
                # test_h_data_list.append(h_vis.cpu().detach().numpy().reshape(h_gcn_train.shape[0], -1))

                if args.name == 'SWaT':
                    test_h_label_list.append(label.numpy())
                    test_log_prob_list.append(loss)
                    test_idx_list.append(idx.numpy())

        for x, x_aug, label, idx in train_loader:
            x = x.to(device)
            h_vis, log_prob, h_gcn_train = model.test(x, )  

            h_vis = h_vis.to(device_test)
            log_prob = log_prob.to(device_test)
            h_gcn_test = h_gcn_test.to(device_test)

            loss = -log_prob.cpu().numpy()
            loss_train.append(loss)

            if args.log_emb:
                train_input_data_list.append(x.cpu().detach().numpy().reshape(x.shape[0], -1))
                train_h_data_list.append(h_vis.cpu().detach().numpy().reshape(h_gcn_train.shape[0], -1))
                train_h_label_list.append(label.numpy())
                if args.name == 'SWaT':
                    train_log_prob_list.append(loss)
                    train_idx_list.append(idx.numpy())

    loss_test = np.concatenate(loss_test)
    loss_train = np.concatenate(loss_train)
    roc_test = roc_auc_score(np.asarray(test_loader.dataset.label, dtype=int), loss_test)
    ap_test = average_precision_score(np.asarray(test_loader.dataset.label, dtype=int), loss_test)

    if roc_max < roc_test:
        roc_max = roc_test
        ap_max = ap_test

        # save embedding
        if args.log_emb:
            train_input_data = np.concatenate(train_input_data_list, axis=0)
            train_h_data = np.concatenate(train_h_data_list, axis=0)
            train_h_label = np.concatenate(train_h_label_list, axis=0)

            # test_input_data = np.concatenate(test_input_data_list, axis=0)
            # test_h_data = np.concatenate(test_h_data_list, axis=0)
            # test_h_label = np.concatenate(test_h_label_list, axis=0)

            if args.name == 'SWaT':
                train_log_prob = np.concatenate(train_log_prob_list, axis=0)
                print("train_log_prob:", train_log_prob.shape)
                test_log_prob = np.concatenate(test_log_prob_list, axis=0)
                print("test_log_prob:", test_log_prob.shape)

                train_idx = np.concatenate(train_idx_list, axis=0)
                print("train_idx:", train_idx.shape)
                test_idx = np.concatenate(test_idx_list, axis=0)
                print("test_idx:", test_idx.shape)

                test_h_label = np.concatenate(test_h_label_list, axis=0)

                best_feature_data = {
                    'train': {
                        'original': train_input_data,
                        'features': train_h_data,
                        'labels': train_h_label,
                        'log_prob': train_log_prob,
                        'idx': train_idx
                    },
                    'test': {
                        # 'original': test_input_data,
                        # 'features': test_h_data,
                        'labels': test_h_label,
                        'log_prob': test_log_prob,
                        'idx': test_idx
                    }
                }
            else:
                best_feature_data = {
                    'train': {
                        'original': train_input_data,
                        'features': train_h_data,
                        'labels': train_h_label
                    },
                    # 'test': {
                    #     'original': test_input_data,
                    #     'features': test_h_data,
                    #     'labels': test_h_label
                }

            save_Embedding = os.path.join('save_Embedding', args.name)
            if not os.path.exists(save_Embedding):
                os.makedirs(save_Embedding)

            best_feature_data_file_name = f'{save_Embedding}/path_to_{args.name}_dataset.pkl'
            print('best_feature_data_file_name:', best_feature_data_file_name)
            with open(best_feature_data_file_name, 'wb') as file:
                pickle.dump(best_feature_data, file)

        if args.save_model:
            save_model_file_name = f"{save_path}/{args.name}_dataset_model.pth"
            torch.save({'model': model.state_dict()}, save_model_file_name)
            print("save_model_file_name:", save_model_file_name)

        if args.log_loss:
            saved_test_label = np.asarray(test_loader.dataset.label, dtype=int)
            saved_test_loss = {
                'label': saved_test_label,
                'loss_test': loss_test
            }

            saved_loss = os.path.join('saved_loss', args.name)
            if not os.path.exists(saved_loss):
                os.makedirs(saved_loss)

            save_loss_file = f'{saved_loss}/{args.name}_dataset_loss.pkl'
            print('save_loss_file:', save_loss_file)
            with open(save_loss_file, 'wb') as file:
                pickle.dump(saved_test_loss, file)

    print('epoch:', epoch, 'seed:', args.seed, 'roc_max:', roc_max, 'ap_max:', ap_max)

    # wandb.log({
    #     'loss_mani_po': loss_mani_po,
    #     'loss_mani_ne': loss_mani_ne,
    #     'loss_train': np.mean(loss_train),
    #     'loss_test': np.mean(loss_test),
    #     'roc_test_max': roc_max,
    #     'roc_test': roc_test,
    #     'ap_test': ap_test,
    #     'ap_test_max': ap_max
    # })

# wandb.finish()
