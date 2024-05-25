from math import gamma
import os
import argparse
from statistics import mode
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import MultiStepLR

import wandb

from models.DeepSAD import DeepSVDD, DeepSAD
from models.DROCC import DROCCTrainer, LSTM_FC  # DROCC

from models.GAN import R_Net, D_Net, CNNAE, train_model, R_Loss, D_Loss, test_single_epoch
import numpy as np

parser = argparse.ArgumentParser()
# files
parser.add_argument('--data_dir', type=str,
                    default='Data/input/SWaT_Dataset_Attack_v0.csv', help='Location of datasets.')
parser.add_argument('--output_dir', type=str,
                    default='./checkpoint/model')
parser.add_argument('--name', default='GANF_Water')
# restore
parser.add_argument('--graph', type=str, default='None')
parser.add_argument('--model', type=str, choices=['DeepSVDD', 'DeepSAD', 'DROCC', 'EncDecAD', 'ALOCC', 'DAGMM', 'USAD'],
                    default='None')
parser.add_argument('--seed', type=int, default=18, help='Random seed to use.')
parser.add_argument('--load', type=str, default="")

# made parameters
parser.add_argument('--n_blocks', type=int, default=1,
                    help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--n_components', type=int, default=1, help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--batch_norm', type=bool, default=False)
# training params
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--log_interval', type=int, default=1, help='How often to show loss statistics and save samples.')
parser.add_argument('--window_size', type=int, default=60)
parser.add_argument('--stride_size', type=int, default=10)
parser.add_argument('--train_split', type=float, default=0.6)

args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

wandb.login(
    host='http://10.66.103.228:4080',
    key='local-10402bfd1dd12c77ff954ce9299d674a4708d06d',
)

wandb.init(project="other_model", name=args.name, config=args)

import random
import numpy as np
import math

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

seed = args.seed
# %%
print("Loading dataset")
print(args.name)

from Dataset import load_smd_smap_msl, loader_SWat, loader_WADI, loader_PSM, loader_WADI_OCC

if args.name == 'SWaT':
    train_loader, val_loader, test_loader, n_sensor = loader_SWat(args.data_dir, args.batch_size, args.window_size,
                                                                  args.stride_size,
                                                                  args.train_split)

elif args.name == 'Wadi':
    train_loader, val_loader, test_loader, n_sensor = loader_WADI(args.data_dir, args.batch_size, args.window_size,
                                                                  args.stride_size,
                                                                  args.train_split)

elif args.name == 'SMAP' or args.name == 'MSL' or args.name.startswith('machine'):
    train_loader, val_loader, test_loader, n_sensor = load_smd_smap_msl(args.name, args.batch_size, args.window_size,
                                                                        args.stride_size,
                                                                        args.train_split)

elif args.name == 'PSM':
    train_loader, val_loader, test_loader, n_sensor = loader_PSM(args.name, args.batch_size, args.window_size, args.stride_size,
                                                                 args.train_split)
print("Loading dataset")
size = n_sensor
print("n_sensor:", n_sensor, 'size:', size)

# %%
if args.model == 'DROCC':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    net = LSTM_FC(input_dim=size).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)
    radius = math.sqrt(size) / 2
    gamma = 2
    lam = 0.0001

    if args.load:
        net.load_state_dict(torch.load(args.load)['model'])

    model = DROCCTrainer(net, optimizer, lam, radius, gamma, device)
    roc_test_max, ap_test_max = model.train(args, seed, train_loader=train_loader, test_loader=test_loader,
                                            lr_scheduler=scheduler, total_epochs=args.n_epochs, save_path='./othermodel',
                                            name='DROCC')
    print('roc_test_max:', roc_test_max, 'ap_test_max', ap_test_max)
    # gt, pre = model.test(test_loader)
    # ROC(args, gt, pre)
    wandb.log({
        'roc_test_max': roc_test_max,
        'ap_test_max': ap_test_max
    })

elif args.model == 'DeepSAD':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = DeepSAD(n_sensor, size, device)

    if args.load:
        model.ae_net.encoder.load_state_dict(torch.load(args.load)['model'])
        c = torch.load(args.load)['c']

    roc_test_max, ap_test_max = model.train(train_loader, test_loader, args, device)
    print('roc_test_max:', roc_test_max, 'ap_test_max', ap_test_max)
    # gt, pre = model.test(test_loader, c, 1, device)
    # ROC(args, gt, pre)
    wandb.log({
        'roc_test_max': roc_test_max,
        'ap_test_max': ap_test_max
    })


elif args.model == 'DAGMM':
    from models.dagmm import DAGMM

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    hyp = {
        'input_dim': size * args.window_size,
        'hidden1_dim': 60,
        'hidden2_dim': 30,
        'hidden3_dim': 10,
        'zc_dim': 1,
        'emb_dim': 10,
        'n_gmm': 2,
        'dropout': 0.5,
        'lambda1': 0.1,
        'lambda2': 0.005,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': 10,
        'ratio': 0.8
    }
    # train
    model = DAGMM(hyp)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), hyp['lr'], amsgrad=True)
    scheduler = MultiStepLR(optim, [5, 8], 0.1)
    loss_total = 0
    recon_error_total = 0
    e_total = 0
    roc_max = 0
    ap_max = 0

    for epoch in range(hyp['epochs']):
        model.train()
        for i, (input_data, labels, _) in enumerate(train_loader):
            # import pdb;
            # pdb.set_trace()
            input_data = input_data.to(device)

            optim.zero_grad()

            enc, dec, z, gamma = model(input_data)
            input_data, dec, z, gamma = input_data.cpu(), dec.cpu(), z.cpu(), gamma.cpu()
            loss, recon_error, e, p = model.loss_func(input_data, dec, gamma, z)
            # print('loss',loss,'recon_error',recon_error,'e',e,'p',p)

            loss_total += loss.item()
            recon_error_total += recon_error.item()
            e_total += e.item()

            # model.zero_grad()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optim.step()

        scheduler.step()

        model.eval()
        loss_test = []
        with torch.no_grad():
            for x, y, _ in test_loader:

                enc, dec, z, gamma = model(x)
                m_prob, m_mean, m_cov = model.get_gmm_param(gamma, z)

                for i in range(z.shape[0]):
                    zi = z[i].unsqueeze(1)
                    sample_energy = model.sample_energy(m_prob, m_mean, m_cov, zi, gamma.shape[1], gamma.shape[0])
                    se = sample_energy.detach().item()

                    loss_test.append(se)

        roc_test = roc_auc_score(np.asarray(test_loader.dataset.label, dtype=int), loss_test)
        ap_test = average_precision_score(np.asarray(test_loader.dataset.label, dtype=int), loss_test)

        if roc_max < roc_test:
            roc_max = roc_test
            ap_max = ap_test

        print('epoch:', epoch, 'seed:', args.seed, 'roc_max:', roc_max, 'ap_max:', ap_max)

        wandb.log({
            'roc_test_max': roc_max,
            'ap_test_max': ap_max
        })

    wandb.finish()

elif args.model == 'USAD':
    from models.usad import UsadModel, to_device, training

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    w_size = args.window_size * n_sensor  # 3060
    z_size = args.window_size * args.hidden_size

    model = UsadModel(w_size, z_size)
    model = to_device(model, device)

    roc_max, ap_max = training(args.n_epochs, model, train_loader, test_loader)

    wandb.log({
        'roc_test_max': roc_max,
        'ap_test_max': ap_max
    })

    wandb.finish()
