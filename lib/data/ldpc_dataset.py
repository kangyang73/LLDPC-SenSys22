import torch
from torch.utils.data import Dataset
import numpy as np


class ldpc_graph_structure_generator:
    def __init__(self, check_len_t, code_len_t, ldpc_path_t, degree_row_t, degree_col_t):
        self.check_len = check_len_t
        self.code_len = code_len_t
        self.ldpc_path = ldpc_path_t
        self.degree_row = degree_row_t
        self.degree_col = degree_col_t
        self.ldpc_file = self.ldpc_path

        self.get_factor_structure()

    def get_factor_structure(self):
        n_nodes = self.code_len + self.check_len 
        n_edges = self.degree_row
        nn_idx = np.zeros((n_nodes, n_edges)).astype(np.int64)
        efeature = np.zeros((2, n_nodes, n_edges)).astype(np.float32)

        with open(self.ldpc_file) as f:
            for i in range(4):
                f.readline()

            for i in range(self.code_len):
                te = f.readline().strip().split()
                indice = map(lambda x: int(x)-1, te)
                for j, idx in enumerate(indice):
                    nn_idx[i, j] = self.code_len + idx
                    efeature[0, i, j] = 1

                for k in range(j + 1, n_edges):
                    nn_idx[i, k] = i

            factors = []

            for i in range(self.check_len):
                te2 = f.readline().strip().split()
                indice = map(lambda x: int(x)-1, te2)
                cfactor = []
                for j, idx in enumerate(indice):
                    cfactor.append(idx)
                    nn_idx[self.code_len + i, j] = idx
                    efeature[1, self.code_len + i, j] = 1

                factors.append(cfactor)

            self.factors = np.asarray(factors, np.int64)
            self.etype = efeature
            self.nn_idx = nn_idx


    def get_pairwise_structure(self):
        n_nodes = self.code_len
        n_edges = self.degree_col * self.degree_row
        nn_idx = np.zeros((n_nodes, n_edges)).astype(np.int64)
        efeature = np.zeros((1, n_nodes, n_edges)).astype(np.float32)

        v2f = np.zeros((self.code_len, self.degree_col)).astype(np.int64)
        f2v = np.zeros((self.check_len, self.degree_row)).astype(np.int64)

        with open(self.ldpc_file) as f:
            for i in range(4):
                f.readline()

            for i in range(self.code_len):
                v2f[i, :] = list(
                    map(lambda x: int(x)-1, f.readline().strip().split()))

            for i in range(self.check_len):
                f2v[i, :] = list(
                    map(lambda x: int(x)-1, f.readline().strip().split()))

            for i in range(self.code_len):
                k = 0
                for f in v2f[i]:
                    for v in f2v[f]:
                        nn_idx[i, k] = v
                        efeature[0, i, k] = 1
                        k += 1

            self.factors = f2v
            self.etype = efeature
            self.nn_idx = nn_idx


    def get_highorder_feature(self, y):
        t1 = self.factors.reshape(-1)
        t2 = np.take(y, t1)
        t3 = self.factors.shape
        t4 = t2.reshape(t3)
        return t4


    def get_mpnn_sp_structure(self, y):
        hop = self.get_highorder_feature(y)

        nn_idx_f2v = self.nn_idx[:self.code_len, :self.degree_col] - self.code_len
        nn_idx_v2f = self.nn_idx[self.code_len:, :]

        x1 = nn_idx_f2v.reshape(-1)
        x2 = np.take(hop, x1, axis=0)
        x3 = x2.reshape(self.code_len, self.degree_col, self.degree_row)
        efeature_f2v = x3.astype(np.float32)

        q1 = y.reshape(self.code_len, 1, 1)
        q2 = np.repeat(q1, self.degree_col, axis=1)
        efeature_f2v = np.concatenate([efeature_f2v, q2], axis=2)

        m1 = hop.reshape(self.check_len, 1, self.degree_row)
        efeature_v2f = np.repeat(m1, self.degree_row, axis=1)

        m2 = hop.reshape(self.check_len, self.degree_row, 1)
        efeature_v2f = np.concatenate((efeature_v2f, m2), axis=2)

        return hop, nn_idx_f2v, nn_idx_v2f, efeature_f2v, efeature_v2f

    def get_high_factor_structure(self, y):
        hop = self.get_highorder_feature(y)

        nn_idx_node = self.nn_idx[:self.code_len, :self.degree_col]
        
        feature_h = np.take(hop, nn_idx_node.reshape(-1) - self.code_len,
                            axis=0).reshape(self.code_len, self.degree_col, self.degree_row).astype(np.float32)

        efeature_node = np.concatenate([feature_h, np.repeat(y.reshape(self.code_len, 1, 1), self.degree_col, axis=1)], axis=2)
        efeature_node_pad = np.zeros_like(efeature_node).astype(np.float32)

        efeature_node = np.concatenate((efeature_node, efeature_node_pad), axis=1)

        efeature_hop = np.repeat(hop.reshape(self.check_len, 1, self.degree_row), self.degree_row, axis=1)
        efeature_hop = np.concatenate((efeature_hop, hop.reshape(self.check_len, self.degree_row, 1)), axis=2)

        efeature = np.concatenate((efeature_node, efeature_hop), axis=0)

        return self.nn_idx, self.etype, efeature, hop


class Codes(Dataset):
    def __init__(self, filename, check_len_t, code_len_t, ldpc_path_t, degree_row_t, degree_col_t):
        self.data = torch.load(filename)

        self.generator = ldpc_graph_structure_generator(check_len_t, 
                                                        code_len_t, 
                                                        ldpc_path_t, 
                                                        degree_row_t, 
                                                        degree_col_t)

    def __len__(self):
        return len(self.data['noizy_sg'])

    def __getitem__(self, idx):
        noizy_sg = self.data['noizy_sg'][idx].numpy()
        orig = self.data['gts'][idx].numpy()

        hop, nn_idx_f2v, nn_idx_v2f, efeature_f2v, efeature_v2f = \
            self.generator.get_mpnn_sp_structure(noizy_sg)
        
        # node_feature = self.data['noizy_sg'][idx].view(-1, 1).numpy().T
        node_feature = torch.cat([self.data['noizy_sg'][idx].view(-1, 1), 
                                  self.data['snrs'][idx].view(-1, 1)], dim=1).numpy().T

        hop_feature = np.expand_dims(hop.T, -1).astype(np.float32)
        efeature_v2f = np.transpose(efeature_v2f, [2, 0, 1]).astype(np.float32)
        efeature_f2v = np.transpose(efeature_f2v, [2, 0, 1]).astype(np.float32)

        return node_feature, \
            hop_feature, \
                nn_idx_f2v.astype(np.int32), \
                    nn_idx_v2f.astype(np.int32), \
                        efeature_f2v, \
                            efeature_v2f, \
                                orig.astype(np.int32)
    



