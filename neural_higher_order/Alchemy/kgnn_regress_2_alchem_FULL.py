from __future__ import division

import sys

import auxiliarymethods.datasets as dp
import kernel_baselines as kb

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import os.path as osp
import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool, GINConv,Set2Set

from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader
import torch.nn.functional as F


class ZINC(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(ZINC, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "ZINC_adllrrdrdvv10kd"

    @property
    def processed_file_names(self):
        return "ZINC_alrldrrvv1dd0kd"

    def download(self):
        pass

    def process(self):
        data_list = []

        indices_train = []
        indices_val = []
        indices_test = []

        infile = open("data/test_al_50.index", "r")
        for line in infile:
            indices_test = line.split(",")
            indices_test = [int(i) for i in indices_test]

        infile = open("data/val_al_50.index", "r")
        for line in infile:
            indices_val = line.split(",")
            indices_val = [int(i) for i in indices_val]

        infile = open("data/train_al_50.index", "r")
        for line in infile:
            indices_train = line.split(",")
            indices_train = [int(i) for i in indices_train]

        print("###")

        targets = dp.get_dataset("alchemy_full", multigregression=True)
        tmp1 = targets[indices_train].tolist()
        tmp2 = targets[indices_val].tolist()
        tmp3 = targets[indices_test].tolist()
        targets = tmp1
        targets.extend(tmp2)
        targets.extend(tmp3)

        print("###")
        node_labels = kb.get_all_node_labels_allchem(True, True, indices_train, indices_val, indices_test)

        print(len(targets))

        print("###")
        matrices = kb.get_all_matrices("alchemy_full", indices_train)
        matrices.extend(kb.get_all_matrices("alchemy_full", indices_val))
        matrices.extend(kb.get_all_matrices("alchemy_full", indices_test))


        print(len(matrices))

        for i, m in enumerate(matrices):
            print(i)
            edge_index_1 = torch.tensor(matrices[i][0]).t().contiguous()
            edge_index_2 = torch.tensor(matrices[i][1]).t().contiguous()

            data = Data()
            data.edge_index_1 = edge_index_1
            data.edge_index_2 = edge_index_2

            one_hot = np.eye(83)[node_labels[i]]
            data.x = torch.from_numpy(one_hot).to(torch.float)
            data.y = data.y = torch.from_numpy(np.array([targets[i]])).to(torch.float)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MyData(Data):
    def __inc__(self, key, value):
        return self.num_nodes if key in [
            'edge_index_1', 'edge_index_2'
        ] else 0


class MyTransform(object):
    def __call__(self, data):
        new_data = MyData()
        for key, item in data:
            new_data[key] = item
        return new_data


class NetGIN(torch.nn.Module):
    def __init__(self, dim):
        super(NetGIN, self).__init__()

        num_features = 83

        nn1_1 = Sequential(Linear(num_features, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn1_2 = Sequential(Linear(num_features, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv1_1 = GINConv(nn1_1, train_eps=True)
        self.conv1_2 = GINConv(nn1_2, train_eps=True)
        self.mlp_1 = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn2_1 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn2_2 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv2_1 = GINConv(nn2_1, train_eps=True)
        self.conv2_2 = GINConv(nn2_2, train_eps=True)
        self.mlp_2 = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn3_1 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn3_2 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv3_1 = GINConv(nn3_1, train_eps=True)
        self.conv3_2 = GINConv(nn3_2, train_eps=True)
        self.mlp_3 = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn4_1 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn4_2 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv4_1 = GINConv(nn4_1, train_eps=True)
        self.conv4_2 = GINConv(nn4_2, train_eps=True)
        self.mlp_4 = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn5_1 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn5_2 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv5_1 = GINConv(nn5_1, train_eps=True)
        self.conv5_2 = GINConv(nn5_2, train_eps=True)
        self.mlp_5 = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn6_1 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn6_2 = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv6_1 = GINConv(nn6_1, train_eps=True)
        self.conv6_2 = GINConv(nn6_2, train_eps=True)
        self.mlp_6 = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())
        self.set2set = Set2Set(1 * dim, processing_steps=6)
        self.fc1 = Linear(2 * dim, dim)
        self.fc4 = Linear(dim, 12)

    def forward(self, data):
        x = data.x

        x_1 = F.relu(self.conv1_1(x, data.edge_index_1))
        x_2 = F.relu(self.conv1_2(x, data.edge_index_2))
        x_1_r = self.mlp_1(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv2_1(x_1_r, data.edge_index_1))
        x_2 = F.relu(self.conv2_2(x_1_r, data.edge_index_2))
        x_2_r = self.mlp_2(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv3_1(x_2_r, data.edge_index_1))
        x_2 = F.relu(self.conv3_2(x_2_r, data.edge_index_2))
        x_3_r = self.mlp_3(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv4_1(x_3_r, data.edge_index_1))
        x_2 = F.relu(self.conv4_2(x_3_r, data.edge_index_2))
        x_4_r = self.mlp_4(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv5_1(x_4_r, data.edge_index_1))
        x_2 = F.relu(self.conv5_2(x_4_r, data.edge_index_2))
        x_5_r = self.mlp_5(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv6_1(x_5_r, data.edge_index_1))
        x_2 = F.relu(self.conv6_2(x_5_r, data.edge_index_2))
        x_6_r = self.mlp_6(torch.cat([x_1, x_2], dim=-1))

        x = x_6_r

        x = self.set2set(x, data.batch)

        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        return x



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'ZINC')
dataset = ZINC(path, transform=MyTransform())
print(len(dataset))


print("###")
mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean.to(device), std.to(device)


train_dataset = dataset[0:50000].shuffle()
# mean = train_dataset.data.y.mean(dim=0, keepdim=True)
# std = train_dataset.data.y.std(dim=0, keepdim=True)
# train_dataset.data.y = (train_dataset.data.y - mean) / std

val_dataset = dataset[50000:55000].shuffle()
# mean = val_dataset.data.y.mean(dim=0, keepdim=True)
# std = val_dataset.data.y.std(dim=0, keepdim=True)
# val_dataset.data.y = (val_dataset.data.y - mean) / std

test_dataset = dataset[55000:60000].shuffle()
# mean = test_dataset.data.y.mean(dim=0, keepdim=True)
# std = test_dataset.data.y.std(dim=0, keepdim=True)
# test_dataset.data.y = (test_dataset.data.y - mean) / std

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


results = []
results_log = []
for _ in range(5):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NetGIN(64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=5,
                                                           min_lr=0.0000001)

    def train():
        model.train()
        loss_all = 0

        lf = torch.nn.L1Loss()
        error = torch.zeros([1, 12]).to(device)
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = lf(model(data), data.y)
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
            with torch.no_grad():
                error += ((data.y * std - model(data) * std).abs() / std).sum(dim=0)
        error = error / len(train_loader.dataset)

        return loss_all / len(train_loader.dataset), error.mean().item()


    @torch.no_grad()
    def test(loader):
        model.eval()
        error = torch.zeros([1, 12]).to(device)

        for data in loader:
            data = data.to(device)
            error += ((data.y * std - model(data) * std).abs() / std).sum(dim=0)

        error = error / len(loader.dataset)
        error_log = torch.log(error)

        return  error.mean().item(), error_log.mean().item()


    best_val_error = None
    test_error = None
    test_error_log = None
    for epoch in range(1, 501):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss, _ = train()
        val_error, _ = test(val_loader)

        scheduler.step(val_error)
        if best_val_error is None or val_error <= best_val_error:
            test_error, test_error_log = test(test_loader)
            best_val_error = val_error

        print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
              'Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error))

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
