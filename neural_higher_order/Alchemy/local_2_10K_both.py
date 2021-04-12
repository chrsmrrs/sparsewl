import sys

sys.path.insert(0, '..')
sys.path.insert(0, '.')

import auxiliarymethods.datasets as dp
import preprocessing as pre

import os.path as osp
import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, Set2Set
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import torch.nn.functional as F


class Alchemy(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(Alchemy, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "alchemy10_both"

    @property
    def processed_file_names(self):
        return "alchemy10_both"

    def download(self):
        pass

    def process(self):
        data_list = []

        indices_train = []
        indices_val = []
        indices_test = []

        infile = open("test_al_10.index", "r")
        for line in infile:
            indices_test = line.split(",")
            indices_test = [int(i) for i in indices_test]

        infile = open("val_al_10.index", "r")
        for line in infile:
            indices_val = line.split(",")
            indices_val = [int(i) for i in indices_val]

        infile = open("train_al_10.index", "r")
        for line in infile:
            indices_train = line.split(",")
            indices_train = [int(i) for i in indices_train]

        targets = dp.get_dataset("alchemy_full", multigregression=True)
        tmp_1 = targets[indices_train].tolist()
        tmp_2 = targets[indices_val].tolist()
        tmp_3 = targets[indices_test].tolist()
        targets = tmp_1
        targets.extend(tmp_2)
        targets.extend(tmp_3)

        node_labels_con = pre.get_all_node_labels_allchem_con(True, True, indices_train, indices_val, indices_test)
        node_labels_unc = pre.get_all_node_labels_allchem(True, True, indices_train, indices_val, indices_test)

        matrices_con = pre.get_all_matrices_con("alchemy_full", indices_train)
        matrices_con.extend(pre.get_all_matrices_con("alchemy_full", indices_val))
        matrices_con.extend(pre.get_all_matrices_con("alchemy_full", indices_test))

        matrices_unc = pre.get_all_matrices_unc("alchemy_full", indices_train)
        matrices_unc.extend(pre.get_all_matrices_unc("alchemy_full", indices_val))
        matrices_unc.extend(pre.get_all_matrices_unc("alchemy_full", indices_test))

        for i, m in enumerate(matrices_con):
            edge_index_0_con = torch.tensor(matrices_con[i][0]).t().contiguous()
            edge_index_1_con = torch.tensor(matrices_con[i][1]).t().contiguous()

            edge_index_0_unc = torch.tensor(matrices_unc[i][0]).t().contiguous()
            edge_index_1_unc = torch.tensor(matrices_unc[i][1]).t().contiguous()

            data = Data()
            data.edge_index_0_con = edge_index_0_con
            data.edge_index_1_con = edge_index_1_con

            data.edge_index_0_unc = edge_index_0_unc
            data.edge_index_1_unc = edge_index_1_unc

            one_hot = np.eye(83)[node_labels_con[i]]
            data.x_con = torch.from_numpy(one_hot).to(torch.float)

            one_hot = np.eye(83)[node_labels_unc[i]]
            data.x_unc = torch.from_numpy(one_hot).to(torch.float)

            data.batch_con = torch.from_numpy(np.zeros(int(edge_index_0_con[0].max().item()) + 1, dtype=np.int64)).to(torch.long)
            data.batch_unc = torch.from_numpy(np.zeros(int(edge_index_0_unc[0].max().item()) + 1, dtype=np.int64)).to(torch.long)

            data.num_con_0 = int(edge_index_0_con[0].max().item()) + 1
            data.num_con_1 = int(edge_index_1_con[1].max().item()) + 1
            data.num_unc_0 = int(edge_index_0_unc[0].max().item()) + 1
            data.num_unc_1 = int(edge_index_1_unc[1].max().item()) + 1

            data.y = torch.from_numpy(np.array([targets[i]])).to(torch.float)

            if edge_index_1_con[0].max() != edge_index_1_con[1].max():
                print("xxx")
                exit()

            if edge_index_1_unc[0].max() != edge_index_1_unc[1].max():
                print("xxx")
                exit()

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MyData(Data):
    def __inc__(self, key, value):
        if key in ['edge_index_0_con']:
            return self.num_con_0
        if key in ['edge_index_1_con']:
            return self.num_con_1
        if key in ['edge_index_0_unc']:
            return self.num_unc_0
        if key in ['edge_index_1_unc']:
            return self.num_unc_1
        if key in ['batch_con']:
            return 1
        if key in ['batch_unc']:
            return 1
        else:
            return 0



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

        nn1_1_unc = Sequential(Linear(num_features, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn1_2_unc = Sequential(Linear(num_features, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv1_1_unc = GINConv(nn1_1_unc, train_eps=True)
        self.conv1_2_unc = GINConv(nn1_2_unc, train_eps=True)
        self.mlp_1_unc = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn2_1_unc = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn2_2_unc = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv2_1_unc = GINConv(nn2_1_unc, train_eps=True)
        self.conv2_2_unc = GINConv(nn2_2_unc, train_eps=True)
        self.mlp_2_unc = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn3_1_unc = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn3_2_unc = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv3_1_unc = GINConv(nn3_1_unc, train_eps=True)
        self.conv3_2_unc = GINConv(nn3_2_unc, train_eps=True)
        self.mlp_3_unc = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn4_1_unc = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn4_2_unc = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv4_1_unc = GINConv(nn4_1_unc, train_eps=True)
        self.conv4_2_unc = GINConv(nn4_2_unc, train_eps=True)
        self.mlp_4_unc = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn5_1_unc = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn5_2_unc = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv5_1_unc = GINConv(nn5_1_unc, train_eps=True)
        self.conv5_2_unc = GINConv(nn5_2_unc, train_eps=True)
        self.mlp_5_unc = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn6_1_unc = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn6_2_unc = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv6_1_unc = GINConv(nn6_1_unc, train_eps=True)
        self.conv6_2_unc = GINConv(nn6_2_unc, train_eps=True)
        self.mlp_6_unc = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())
        self.set2set_unc = Set2Set(1 * dim, processing_steps=6)
        
        ### conncected ###
        nn1_1_con = Sequential(Linear(num_features, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn1_2_con = Sequential(Linear(num_features, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv1_1_con = GINConv(nn1_1_con, train_eps=True)
        self.conv1_2_con = GINConv(nn1_2_con, train_eps=True)
        self.mlp_1_con = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn2_1_con = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn2_2_con = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv2_1_con = GINConv(nn2_1_con, train_eps=True)
        self.conv2_2_con = GINConv(nn2_2_con, train_eps=True)
        self.mlp_2_con = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn3_1_con = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn3_2_con = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv3_1_con = GINConv(nn3_1_con, train_eps=True)
        self.conv3_2_con = GINConv(nn3_2_con, train_eps=True)
        self.mlp_3_con = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn4_1_con = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn4_2_con = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv4_1_con = GINConv(nn4_1_con, train_eps=True)
        self.conv4_2_con = GINConv(nn4_2_con, train_eps=True)
        self.mlp_4_con = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn5_1_con = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn5_2_con = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv5_1_con = GINConv(nn5_1_con, train_eps=True)
        self.conv5_2_con = GINConv(nn5_2_con, train_eps=True)
        self.mlp_5_con = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn6_1_con = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        nn6_2_con = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                           torch.nn.BatchNorm1d(dim), ReLU())
        self.conv6_1_con = GINConv(nn6_1_con, train_eps=True)
        self.conv6_2_con = GINConv(nn6_2_con, train_eps=True)
        self.mlp_6_con = Sequential(Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())
        self.set2set_con = Set2Set(1 * dim, processing_steps=6)

        self.mlp_fuse = Sequential(Linear(4 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        self.fc1 = Linear(1 * dim, dim)
        self.fc4 = Linear(dim, 12)

    def forward(self, data):
        x = data.x_unc

        x_1_r = self.nn1_1_con(x)

        #x_1 = F.relu(self.conv1_1_unc(x, data.edge_index_0_unc))
        #x_2 = F.relu(self.conv1_2_unc(x, data.edge_index_1_unc))
        #x_1_r = self.mlp_1_unc(torch.cat([x_1, x_2], dim=-1))

        # x_1 = F.relu(self.conv2_1_unc(x_1_r, data.edge_index_0_unc))
        # x_2 = F.relu(self.conv2_2_unc(x_1_r, data.edge_index_1_unc))
        # x_2_r = self.mlp_2_unc(torch.cat([x_1, x_2], dim=-1))
        #
        # x_1 = F.relu(self.conv3_1_unc(x_2_r, data.edge_index_0_unc))
        # x_2 = F.relu(self.conv3_2_unc(x_2_r, data.edge_index_1_unc))
        # x_3_r = self.mlp_3_unc(torch.cat([x_1, x_2], dim=-1))
        #
        # x_1 = F.relu(self.conv4_1_unc(x_3_r, data.edge_index_0_unc))
        # x_2 = F.relu(self.conv4_2_unc(x_3_r, data.edge_index_1_unc))
        # x_4_r = self.mlp_4_unc(torch.cat([x_1, x_2], dim=-1))
        #
        # x_1 = F.relu(self.conv5_1_unc(x_4_r, data.edge_index_0_unc))
        # x_2 = F.relu(self.conv5_2_unc(x_4_r, data.edge_index_1_unc))
        # x_5_r = self.mlp_5_unc(torch.cat([x_1, x_2], dim=-1))
        #
        # x_1 = F.relu(self.conv6_1_unc(x_5_r, data.edge_index_0_unc))
        # x_2 = F.relu(self.conv6_2_unc(x_5_r, data.edge_index_1_unc))
        # x_6_r = self.mlp_6_unc(torch.cat([x_1, x_2], dim=-1))

        x = x_1_r

        x_unc = self.set2set_unc(x, data.batch_unc.to(torch.long))

        ### connected ###
        x = data.x_con

        x_1 = F.relu(self.conv1_1_con(x, data.edge_index_0_con))
        x_2 = F.relu(self.conv1_2_con(x, data.edge_index_1_con))
        x_1_r = self.mlp_1_con(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv2_1_con(x_1_r, data.edge_index_0_con))
        x_2 = F.relu(self.conv2_2_con(x_1_r, data.edge_index_1_con))
        x_2_r = self.mlp_2_con(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv3_1_con(x_2_r, data.edge_index_0_con))
        x_2 = F.relu(self.conv3_2_con(x_2_r, data.edge_index_1_con))
        x_3_r = self.mlp_3_con(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv4_1_con(x_3_r, data.edge_index_0_con))
        x_2 = F.relu(self.conv4_2_con(x_3_r, data.edge_index_1_con))
        x_4_r = self.mlp_4_con(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv5_1_con(x_4_r, data.edge_index_0_con))
        x_2 = F.relu(self.conv5_2_con(x_4_r, data.edge_index_1_con))
        x_5_r = self.mlp_5_con(torch.cat([x_1, x_2], dim=-1))

        x_1 = F.relu(self.conv6_1_con(x_5_r, data.edge_index_0_con))
        x_2 = F.relu(self.conv6_2_con(x_5_r, data.edge_index_1_con))
        x_6_r = self.mlp_6_con(torch.cat([x_1, x_2], dim=-1))

        x = x_6_r

        x_con = self.set2set_con(x, data.batch_con.to(torch.long))
        x = self.mlp_fuse(torch.cat([x_con, x_unc], dim=-1))

        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'Alchemy')
dataset = Alchemy(path, transform=MyTransform())

mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean.to(device), std.to(device)

train_dataset = dataset[0:10000].shuffle()
val_dataset = dataset[10000:11000].shuffle()
test_dataset = dataset[11000:12000].shuffle()

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
                                                           factor=0.5, patience=10,
                                                           min_lr=0.0000001)

    def train():
        model.train()
        loss_all = 0

        lf = torch.nn.L1Loss()
        for data in train_loader:
            data = data.to(device)

            optimizer.zero_grad()
            loss = lf(model(data), data.y)

            loss.backward()
            loss_all += loss * batch_size

            optimizer.step()
        return (loss_all / len(train_loader.dataset))

    @torch.no_grad()
    def test(loader):
        model.eval()
        error = torch.zeros([1, 12]).to(device)

        for data in loader:
            data = data.to(device)
            error += ((data.y * std - model(data) * std).abs() / std).sum(dim=0)

        error = error / len(loader.dataset)
        error_log = torch.log(error)

        return error.mean().item(), error_log.mean().item()


    best_val_error = None
    for epoch in range(1, 201):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train()
        val_error, _ = test(val_loader)
        scheduler.step(val_error)

        if best_val_error is None or val_error <= best_val_error:
            test_error, test_error_log = test(test_loader)
            best_val_error = val_error

        print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
              'Test MAE: {:.7f},Test MAE: {:.7f}, '.format(epoch, lr, loss, val_error, test_error, test_error_log))

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
