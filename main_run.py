import argparse
import datetime
import numpy as np
import os
import sys
import torch
import time
from tqdm import tqdm
import lib
from lib.model.mpnn import FactorNN
from utils.types import str2bool, to_cuda
import json
import glob


def find_latest_model(model_path_dir, model_name):
    # Find all FactorNN_ directories and sort by date
    dirs = sorted(glob.glob(os.path.join(model_path_dir, f"{model_name}_*")), 
                  key=lambda x: datetime.datetime.strptime(x.split('_')[-1], "%m%d%H%M"), 
                  reverse=True)

    if dirs:
        # Find the model with the highest epoch in the most recent directory
        models = sorted(glob.glob(os.path.join(dirs[0], f"{model_name}_epoches_*.pt")), 
                        key=lambda x: int(x.split('_')[-1].split('.')[0]), 
                        reverse=True)

        if models:
            return models[0]
        
    else:
        return None


class LDPCModel(torch.nn.Module):
    def __init__(self, nfeature_dim, hop_order, nedge_type, check_len_t, code_len_t, with_residual=True, aggregator='max'):
        super(LDPCModel, self).__init__()

        self.check_len = check_len_t
        self.code_len = code_len_t

        self.main = FactorNN(nfeature_dim, 
                             [hop_order, code_len_t],
                             [64, 64, 64, 128, 256, 256, 128, 64, 64],
                             [nedge_type, 1],
                             1,
                             ret_high=False,
                             aggregator=aggregator)

        self.emodel_f2v = torch.nn.Sequential(torch.nn.Conv2d(hop_order + 1, 64, 1),
                                              torch.nn.ReLU(inplace=True),
                                              torch.nn.Conv2d(64, nedge_type, 1))

        self.emodel_v2f = torch.nn.Sequential(torch.nn.Conv2d(hop_order + 1, 64, 1),
                                              torch.nn.ReLU(inplace=True),
                                              torch.nn.Conv2d(64, nedge_type, 1))

        hetype_v2f = np.ones([1, 1, 1, code_len_t]).astype(np.float32)
        hetype_f2v = np.ones([1, 1, code_len_t, 1]).astype(np.float32)

        hnn_idx_v2f = np.asarray(list(range(code_len_t)), dtype=np.int64).reshape((1, 1, code_len_t))
        hnn_idx_f2v = np.asarray([0] * code_len_t, dtype=np.int64).reshape((1, code_len_t, 1))

        self.hnn_idx_v2f = torch.nn.Parameter(torch.from_numpy(hnn_idx_v2f).long(), requires_grad=False)
        self.hnn_idx_f2v = torch.nn.Parameter(torch.from_numpy(hnn_idx_f2v).long(), requires_grad=False)

        self.hetype_v2f = torch.nn.Parameter(torch.from_numpy(hetype_v2f), requires_grad=False)
        self.hetype_f2v = torch.nn.Parameter(torch.from_numpy(hetype_f2v), requires_grad=False)

        self.with_residual = with_residual

        self.nhop_regressor = torch.nn.Sequential(torch.nn.Linear(64, 128),
                                                  torch.nn.BatchNorm1d(128),
                                                  torch.nn.ReLU(),
                                                  torch.nn.Linear(128, 128),
                                                  torch.nn.ReLU(),
                                                  torch.nn.Linear(128, 1),
                                                  torch.nn.ReLU())

    def forward(self, node_feature, hop_feature, nn_idx_f2v, nn_idx_v2f, efeature_f2v, efeature_v2f):

        etype_f2v = self.emodel_f2v(efeature_f2v)  # torch.Size([32, 4, 384, 2])
        etype_v2f = self.emodel_v2f(efeature_v2f)  # torch.Size([32, 4, 128, 6])

        with torch.no_grad():
            bsize = node_feature.shape[0]
            nhop_feature = node_feature.reshape((bsize, self.code_len, -1, 1))

        res = self.main(node_feature,
                       [hop_feature, nhop_feature],
                       [nn_idx_f2v, self.hnn_idx_f2v.repeat(bsize, 1, 1)],
                       [nn_idx_v2f, self.hnn_idx_v2f.repeat(bsize, 1, 1)],
                       [etype_f2v, self.hetype_f2v.repeat(bsize, 1, 1, 1)],
                       [etype_v2f, self.hetype_v2f.repeat(bsize, 1, 1, 1)])

        if self.with_residual:
            res = res + node_feature[:, :1, :, :]

        batch_size = res.shape[0]

        res = res.squeeze()
        if batch_size == 1:
            res = res.unsqueeze(0)

        return res[:, :(self.code_len - self.check_len)].contiguous()


def parse_args(base_code_folder):
    parser = argparse.ArgumentParser()
    with open(os.path.join(base_code_folder, "configure.json"), 'r') as jsonfile:
        josn_dict = json.load(jsonfile)

    parser.add_argument('--sf',
                        type=int,
                        default=josn_dict["sf"],
                        help="sf")

    parser.add_argument('--n_epochs',
                        type=int,
                        default=josn_dict["n_epochs"],
                        help="training epoches")

    parser.add_argument('--model_name',
                        type=str,
                        default=josn_dict["model_name"],
                        help="model name")

    parser.add_argument('--use_cuda',
                        type=str2bool,
                        default=josn_dict["use_cuda"],
                        help="Use cuda or not")

    parser.add_argument('--snr', type=int, default=josn_dict["snr"], help="snr")

    parser.add_argument('--data_path',
                        type=str,
                        default=josn_dict["data_path"],
                        help="path of the dataset")

    parser.add_argument('--batch_size', 
                        type=int, default=josn_dict["batch_size"])
    
    parser.add_argument('--aggregator', 
                        type=str, default=josn_dict["aggregator"])

    parser.add_argument('--check_len', type=int, 
                        default=josn_dict["check_len"], 
                        help="the number of rows of parity check matrix H")
    
    parser.add_argument('--code_len', type=int, 
                        default=josn_dict["code_len"], 
                        help="the length of transmitted packet")
    
    parser.add_argument('--ldpc_path', type=str, 
                        default=josn_dict["ldpc_path"])
    
    parser.add_argument('--degree_row', type=int, 
                        default=josn_dict["degree_row"], 
                        help="the number of 1 at each row")
    
    parser.add_argument('--degree_col', type=int, 
                        default=josn_dict["degree_col"], 
                        help="he number of 1 at each column")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.use_cuda = False

    return args


def worker_init_fn(idx):
    t = int(time.time() * 1000.0) + idx
    seed = ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) << 8) + ((t & 0x000000ff) << 24)
    np.random.seed(seed)


def model_train(args, model, model_dir, base_code_dir, dataset_base_dir):

    train_path_f = os.path.join(dataset_base_dir, f"sf{args.sf}", f"snr{args.snr}", args.data_path)

    train_dataset = lib.data.Codes(train_path_f, 
                                   args.check_len, 
                                   args.code_len,
                                   os.path.join(base_code_dir, "codes", args.ldpc_path), 
                                   args.degree_row, 
                                   args.degree_col)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=8,
                                               worker_init_fn=worker_init_fn)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2,  weight_decay=1e-8)


    def lr_sched(x, start=10):
        if x <= start:
            return max(1e-2, (1.0 / start) * x)
        else:
            return max(0.99 ** (x - start), 1e-6)


    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_sched(x))

    start_epoch = 0
    gcnt = 0

    def get_model_dict():
        return {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_sche': scheduler.state_dict(),
            'epoch': epoch,
            'gcnt': gcnt
        }

    def get_model_path():
        return os.path.join(model_dir, '{}_epoches_{}.pt'.format(args.model_name, epoch))

    print('\nTRAINING START!\n')

    for epoch in tqdm(range(start_epoch, args.n_epochs)):

        for bcnt, (node_feature, hop_feature, nn_idx_f2v, nn_idx_v2f, efeature_f2v, efeature_v2f, label) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            if args.use_cuda:
                node_feature, hop_feature, nn_idx_f2v, nn_idx_v2f, efeature_f2v, efeature_v2f, label = to_cuda(
                    node_feature, hop_feature, nn_idx_f2v, nn_idx_v2f, efeature_f2v, efeature_v2f, label.float())
            
            if len(node_feature.shape) == 3:
                node_feature = node_feature.unsqueeze(-1)

            pred = model(node_feature, hop_feature, nn_idx_f2v, nn_idx_v2f, efeature_f2v, efeature_v2f)

            label = label[:, :(args.code_len - args.check_len)].contiguous()

            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred.view(-1), label.view(-1).float())
            loss.backward()

            optimizer.step()
            gcnt += 1

        scheduler.step()

        torch.save(get_model_dict(), get_model_path())


def main(nfeature_dim, nedge_types, base_code_dir, dataset_base_dir, output_dir):
    args = parse_args(base_code_dir)

    hop_order = args.degree_row

    model = LDPCModel(nfeature_dim, 
                      hop_order, 
                      nedge_types, 
                      args.check_len, 
                      args.code_len, 
                      aggregator=args.aggregator)

    if args.use_cuda:
        model.cuda()

    subdir = f'{args.model_name}_{datetime.datetime.now().strftime("%m%d%H%M")}'

    model_dir = os.path.join(output_dir, "model_ldpc", f"sf{args.sf}", f"snr{args.snr}", subdir)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("\nTRAIN MODE\n")
    model_train(args, model, model_dir, base_code_dir, dataset_base_dir)


if __name__ == '__main__':

    base_dir = "/home/kangyang/papers/paper7_lldpc/github"

    base_code_dir_t = os.path.join(base_dir, "lldpc_code")
    
    data_log_base_dir_t = os.path.join(base_dir, "data_log")
    if not os.path.exists(data_log_base_dir_t):
            os.makedirs(data_log_base_dir_t)

    dataset_base_dir_t = os.path.join(data_log_base_dir_t, "data")
    output_dir_t = os.path.join(data_log_base_dir_t, "output")

    nfeature_dim_m = 2
    nedge_types_m = 4

    main(nfeature_dim_m, nedge_types_m, base_code_dir_t, dataset_base_dir_t, output_dir_t)

