import os.path as osp
from datetime import datetime
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Batch
from torch_geometric.datasets import QM9

start = datetime.now()

epochs = 1000
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
keep_data_on_gpu = torch.cuda.is_available()
target = None
in_features, out_features = 4 + 13 + 3 + 2, 12

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'eee-QM9')
dataset = QM9(path)

if target is not None:
    print(f'Target: {target}')
    out_features = 1
    dataset.data.y = dataset.data.y[:, target].unsqueeze(1)
else:
    dataset.data.y = dataset.data.y[:, :12]


def get_data_loader(dataset, batch_size, keep_on_gpu=True):
    data = preprocess_graphs(dataset, keep_on_gpu)
    data = PyGDenseDataset(data)
    n_nodes = np.array([g.x.shape[0] for g in dataset])  # number of nodes for each graph
    loader = torch.utils.data.DataLoader(data, batch_sampler=PyGDenseBatchSampler(n_nodes, batch_size))

    return loader


class PyGDenseDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        graph, label = self.data[item]
        return graph, label


def preprocess_graphs(dataset, keep_on_gpu):
    n_feats = dataset.data.x.shape[1]  # number of node features
    pos_feats = dataset.data.pos.shape[1] if dataset.data.pos is not None else 0  # number of pos features
    e_feats = dataset.data.edge_attr.shape[1]  # number of edge features

    out_list = []
    for graph in dataset:
        label = graph.y.squeeze(0)
        matrix = graph_to_mat(graph, n_feats, e_feats, pos_feats)
        if keep_on_gpu:
            matrix, label = matrix.cuda(), label.cuda()
        out_list.append((matrix, label))

    return out_list


def graph_to_mat(g, n_feats, e_feats, pos_feats):
    node_data = g.x  # (n, n_feats)
    edge_data = g.edge_attr  # (|e|, e_feats)
    pos_data = g.pos  # (n, pos_feats) or None
    n = node_data.shape[0]
    diag_indices = torch.arange(n)
    g_mat = torch.zeros((2 + n_feats + pos_feats + e_feats, n, n), dtype=torch.float)

    src, dst = g.edge_index
    g_mat[0, src, dst] = 1  # adj matrix
    g_mat[1, :, :] = torch.eye(n)  # eye matrix
    if g.num_edges:
        g_mat[2:2 + e_feats, src, dst] = edge_data.transpose(0, 1)  # edge features
    g_mat[2 + e_feats:2 + e_feats + n_feats, diag_indices, diag_indices] = node_data.transpose(0, 1)  # node features
    if pos_feats:
        g_mat[2 + e_feats + n_feats:, diag_indices, diag_indices] = pos_data.transpose(0, 1)  # positional data
    return g_mat


class PyGDenseBatchSampler(torch.utils.data.Sampler):
    def __init__(self, n_nodes_array, batch_size):
        """
        Initialization
        :param n_nodes_array: array of sizes of the graphs
        :param batch_size: batch size
        """
        super().__init__(n_nodes_array.size)

        self.dataset_size = n_nodes_array.size
        self.batch_size = batch_size

        self.index_to_batch = {}
        self.node_size_idx = {}
        running_idx = -1

        for n_nodes_i in set(n_nodes_array):

            if n_nodes_i < 1:
                continue
            self.node_size_idx[n_nodes_i] = np.where(n_nodes_array == n_nodes_i)[0]

            n_of_size = len(self.node_size_idx[n_nodes_i])
            n_batches = max(n_of_size / self.batch_size, 1)

            self.node_size_idx[n_nodes_i] = np.array_split(np.random.permutation(self.node_size_idx[n_nodes_i]),
                                                           n_batches)
            for batch in self.node_size_idx[n_nodes_i]:
                running_idx += 1
                self.index_to_batch[running_idx] = batch

        self.n_batches = running_idx + 1

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        batch_order = np.random.permutation(np.arange(self.n_batches))
        for i in batch_order:
            yield self.index_to_batch[i]


def sparse_to_dense(in_batch):
    """
    Transforms a batch in a sparse representation to a dense batch
    :param in_batch: batch of type torch_geometric.data.Batch
    :return: tuple of:
        1. Batch of shape (B,max(N_i),max(N_i), node_feats+pos_feats+edge_feats+1)
        2. y: Targets of shape (B, #num_targets)
        3. node_bools: Boolean tensor of shape (B, N_max). For each batch and node index, indicate whether it
    is a real or padded node
    """
    # SPARSE TO DENSE
    # Node, Edge given features

    d = in_batch.x.device
    # so to get also adj matrix
    edge_attr = torch.cat([in_batch.edge_attr, torch.ones((in_batch.edge_attr.shape[0], 1), device=d)], dim=-1)
    node_attr = in_batch.x if in_batch.pos is None else torch.cat([in_batch.x, in_batch.pos], axis=1)

    node_features, node_bools = torch_geometric.utils.to_dense_batch(node_attr, batch=in_batch.batch)
    edge_feats = torch_geometric.utils.to_dense_adj(in_batch.edge_index, batch=in_batch.batch, edge_attr=edge_attr)
    node_feats_as_adj = node_features.permute(0, 2, 1).diag_embed().permute(0, 2, 3, 1)

    out = torch.cat([node_feats_as_adj, edge_feats], axis=-1)
    return out, in_batch.y, node_bools


def dense_to_sparse(in_batch, y, node_bools=None):
    """
    Transforms a dense graphs batch to a torch_geometric.data.Batch
    :param in_batch: Dense graph batch of shape (B,N_max, N_max, C)
    :param y: Targets of shape (B, #num_targets)
    :param node_bools: Boolean tensor of shape (B, N_max). For each batch and node index, indicate whether it
    is a real or padded node
    :return: batch of type torch_geometric.data.Batch
    """
    # DENSE TO SPARSE, option 1
    assert len(in_batch.shape) == 4, 'Batch should have shape (B,N_max,N_max,C)'
    B, N, _N, C = in_batch.shape
    assert N == _N, 'Batch should have shape (B,N_max,N_max,C)'

    d = in_batch.device

    if node_bools is not None:
        N_i = node_bools.sum(1)  # N_i for each instance in batch
    else:
        N_i = [N] * B

    # get batch tensor
    batch = torch.tensor(list(chain(*[[i] * N_i[i] for i in range(B)])), dtype=torch.long,
                         device=d)  # shape (B*\Sigma(N_i),). For each node - its graph's index in batch.

    # get x tensor (node features)
    diag_features = in_batch.diagonal(dim1=1, dim2=2).permute(0, 2, 1)  # Shape B, N_max, C
    indices = torch.tensor(list(chain(*[list(range(N_i[i])) for i in range(B)])),
                           dtype=torch.long, device=d)  # shape (B*\Sigma(N_i),). For each node - its index in graph.
    x = diag_features[batch, indices]  # Shape (B*\Sigma(N_i),). Node features

    # get edge_index tensor, edge_attr tensor
    start_node = 0
    src = []
    dst = []
    vals = []
    for i, n_i in enumerate(N_i):
        t1 = torch.arange(n_i, device=d).repeat(n_i, 1).t().flatten()
        t2 = torch.arange(n_i, device=d).repeat(n_i)
        src_i = t1[t1 != t2]
        dst_i = t2[t1 != t2]
        vals_i = in_batch[i, src_i, dst_i]  # should be shape (N_i*(N_i-1)/2, C)
        src.append(src_i + start_node)
        dst.append(dst_i + start_node)
        vals.append(vals_i)
        start_node += n_i

    src, dst = torch.cat(src), torch.cat(dst)
    edge_attr = torch.cat(vals)  # shape (\Sigma(N_i*(N_i-1)/2), C)
    edge_index = torch.stack([src, dst], dim=1).t()  # shape (2, \Sigma(N_i*(N_i-1)/2))

    new_batch = Batch(batch=batch, edge_attr=edge_attr, edge_index=edge_index, x=x, y=y)
    return new_batch


def diag_offdiag_maxpool(input):
    x_pool = torch_geometric.nn.max_pool_x(input.batch, input.x, input.batch)[0]  # Shape (B,C)
    edge_batch = input.batch[input.edge_index[0]]
    e_pool = torch_geometric.nn.max_pool_x(edge_batch, input.edge_attr, edge_batch)[0]  # Shape (B,C)
    pooled = torch.cat([x_pool, e_pool], dim=1)  # Shape (B,2C)

    return pooled


def diag_offdiag_maxpool_dense(input, node_bools):
    B, C, N, _ = input.shape
    N_i = node_bools.sum(dim=1)  # B,

    min_val = input.min()
    diag = torch.diagonal(input, dim1=-2, dim2=-1)  # B,C,N
    diag_reduce = torch.diag_embed(diag - min_val)  # B,C,N,N of diagonal matrices
    edges = input - diag_reduce

    max_nodes = []
    max_edges = []
    for i in range(B):
        max_nodes.append(diag[i, :, node_bools[i]].max(dim=1)[0])  # max over only real nodes
        max_edges.append(edges[i, :, :N_i[i], :N_i[i]].max(dim=2)[0].max(dim=1)[0])  # max over edges between real nodes

    max_nodes = torch.stack(max_nodes, dim=0)  # B,C
    max_edges = torch.stack(max_edges, dim=0)  # B,C

    return torch.cat((max_nodes, max_edges), dim=1)  # output B,2C


def diag_offdiag_maxpool_dense2(input):
    N = input.shape[-1]

    max_diag = torch.max(torch.diagonal(input, dim1=-2, dim2=-1), dim=2)[0]  # B,C  maximum over nodes

    # with torch.no_grad():
    max_val = torch.max(max_diag)
    min_val = torch.max(-1 * input)
    val = torch.abs(torch.add(max_val, min_val))

    min_mat = torch.mul(val, torch.eye(N, device=input.device)).view(1, 1, N,
                                                                     N)  # make node values too small to affect max operation

    max_offdiag = torch.max(torch.max(input - min_mat, dim=3)[0], dim=2)[0]  # B,C maximum over edges

    return torch.cat((max_diag, max_offdiag), dim=1)  # output B,2C


class Block(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.mlp1 = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, kernel_size=1, padding=0, bias=True),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, kernel_size=1, padding=0, bias=True),
            nn.ReLU()
        )

        self.skip = nn.Conv2d(in_features + out_features, out_features, kernel_size=1, padding=0, bias=True)

    def forward(self, inputs):
        mlp1 = self.mlp1(inputs)
        mlp2 = self.mlp2(inputs)

        mult = torch.matmul(mlp1, mlp2)

        out = self.skip(torch.cat((inputs, mult), dim=1))
        return out


class Net(nn.Module):
    def __init__(self, in_feats, num_outputs):
        """
        Build the model computation graph, until scores/values are returned at the end
        """
        super().__init__()
        block_features = [400, 400, 400]

        # First part - sequential mlp blocks
        self.reg_blocks = nn.ModuleList()
        for next_layer_features in block_features:
            mlp_block = Block(in_feats, next_layer_features)
            self.reg_blocks.append(mlp_block)
            in_feats = next_layer_features

        # Second part
        self.suffix = nn.Sequential(
            nn.Linear(2 * block_features[-1], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )

    def forward(self, input):
        # x, y, node_bools = sparse_to_dense(input)
        # x = x.permute(0, 3, 1, 2)  # Shape (B,C,N,N)
        x, y = input
        x, y = x.to(device), y.to(device)

        # equivariant part
        for i, block in enumerate(self.reg_blocks):
            x = block(x)

        # pooling layer
        # sparse_x = dense_to_sparse(x.permute(0, 2, 3, 1), y, node_bools)
        # pooled_x = diag_offdiag_maxpool(sparse_x)
        # pooled_x = diag_offdiag_maxpool_dense(x, node_bools)
        pooled_x = diag_offdiag_maxpool_dense2(x)  # B,2C

        # fully connected
        scores = self.suffix(pooled_x)

        return scores


results = []
results_log = []
for _ in range(5):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', '1t-QdM9')


    dataset = QM9(path)

    dataset.data.y = dataset.data.y[:, 0:12]
    dataset = dataset.shuffle()

    tenpercent = int(len(dataset) * 0.1)
    print("###")
    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean.to(device), std.to(device)

    print("###")
    test_dataset = dataset[:tenpercent].shuffle()
    val_dataset = dataset[tenpercent:2 * tenpercent].shuffle()
    train_dataset = dataset[2 * tenpercent:].shuffle()

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    batch_size = 64

    test_loader = get_data_loader(test_dataset, batch_size, keep_on_gpu=keep_data_on_gpu)
    val_loader = get_data_loader(val_dataset, batch_size, keep_on_gpu=keep_data_on_gpu)
    train_loader = get_data_loader(train_dataset, batch_size, keep_on_gpu=keep_data_on_gpu)

    model = Net(in_features, out_features).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    def train():
        model.train()
        loss_all = 0

        lf = torch.nn.L1Loss()
        error = torch.zeros([1, 12]).to(device)

        c = 0
        for data in train_loader:
            _, y = data
            y = y.to(device)

            optimizer.zero_grad()
            out = model(data)
            loss = lf(out, y)
            loss.backward()
            loss_all += loss.item() * y.size(0)
            optimizer.step()

            c += y.size(0)

            with torch.no_grad():
                error += ((y * std - out * std).abs() / std).sum(dim=0)
        error = error / c

        return loss_all / c, error.mean().item()


    @torch.no_grad()
    def test(loader):
        model.eval()
        error = torch.zeros([1, 12]).to(device)

        c = 0
        for data in loader:
            _, y = data
            c += y.size(0)
            error += ((y * std - model(data) * std).abs() / std).sum(dim=0)

        error = error / c
        error_log = torch.log(error)


        return error.mean().item(), error_log.mean().item()


    best_val_error = None
    test_error = None
    test_error_log = None
    for epoch in range(1, 1001):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss, train_error = train()
        val_error, _ = test(val_loader)
        scheduler.step(val_error)

        if best_val_error is None or val_error <= best_val_error:
            test_error, test_error_log = test(test_loader)

            best_val_error = val_error

        print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
              'Test MAE: {:.7f},Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error, test_error_log))

        if lr < 0.000001:
            print("Converged.")
            break

    results.append(test_error)
    results_log.append(test_error_log)

print("########################")
print(results)
results = np.array(results)
print(results.mean(), results.std())

print(results_log)
results_log = np.array(results_log)
print(results_log.mean(), results_log.std())
