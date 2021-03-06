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


class Alchemy_malkin(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(Alchemy_malkin, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "alchemymalkin50"

    @property
    def processed_file_names(self):
        return "alchemymalkin50"

    def download(self):
        pass

    def process(self):
        data_list = []

        indices_train = []
        indices_val = []
        indices_test = []

        infile = open("test_al_50.index", "r")
        for line in infile:
            indices_test = line.split(",")
            indices_test = [int(i) for i in indices_test]

        infile = open("val_al_50.index", "r")
        for line in infile:
            indices_val = line.split(",")
            indices_val = [int(i) for i in indices_val]

        infile = open("train_al_50.index", "r")
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

        node_labels = pre.get_all_node_labels_allchem(True, True, indices_train, indices_val, indices_test)
        matrices = pre.get_all_matrices_dwl("alchemy_full", indices_train)
        matrices.extend(pre.get_all_matrices_dwl("alchemy_full", indices_val))
        matrices.extend(pre.get_all_matrices_dwl("alchemy_full", indices_test))

        for i, m in enumerate(matrices):
            edge_index_1_l = torch.tensor(matrices[i][0]).t().contiguous()
            edge_index_1_g = torch.tensor(matrices[i][1]).t().contiguous()
            edge_index_2_l = torch.tensor(matrices[i][2]).t().contiguous()
            edge_index_2_g = torch.tensor(matrices[i][3]).t().contiguous()

            data = Data()
            data.edge_index_1_l = edge_index_1_l
            data.edge_index_1_g = edge_index_1_g
            data.edge_index_2_l = edge_index_2_l
            data.edge_index_2_g = edge_index_2_g

            one_hot = np.eye(83)[node_labels[i]]
            data.x = torch.from_numpy(one_hot).to(torch.float)
            data.y = data.y = torch.from_numpy(np.array([targets[i]])).to(torch.float)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MyData(Data):
    def __inc__(self, key, value):
        return self.num_nodes if key in [
            'edge_index_1_l', 'edge_index_1_g', 'edge_index_2_l', 'edge_index_2_g'
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

        nn1_1_l = Sequential(Linear(num_features, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        nn1_2_l = Sequential(Linear(num_features, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        nn1_1_g = Sequential(Linear(num_features, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        nn1_2_g = Sequential(Linear(num_features, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        self.conv1_1_l = GINConv(nn1_1_l, train_eps=True)
        self.conv1_2_l = GINConv(nn1_2_l, train_eps=True)
        self.conv1_1_g = GINConv(nn1_1_g, train_eps=True)
        self.conv1_2_g = GINConv(nn1_2_g, train_eps=True)
        self.mlp_1 = Sequential(Linear(4 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())
        nn2_1_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        nn2_2_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        nn2_1_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        nn2_2_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        self.conv2_1_l = GINConv(nn2_1_l, train_eps=True)
        self.conv2_2_l = GINConv(nn2_2_l, train_eps=True)
        self.conv2_1_g = GINConv(nn2_1_g, train_eps=True)
        self.conv2_2_g = GINConv(nn2_2_g, train_eps=True)
        self.mlp_2 = Sequential(Linear(4 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn3_1_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        nn3_2_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        nn3_1_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        nn3_2_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        self.conv3_1_l = GINConv(nn3_1_l, train_eps=True)
        self.conv3_2_l = GINConv(nn3_2_l, train_eps=True)
        self.conv3_1_g = GINConv(nn3_1_g, train_eps=True)
        self.conv3_2_g = GINConv(nn3_2_g, train_eps=True)
        self.mlp_3 = Sequential(Linear(4 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn4_1_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        nn4_2_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        nn4_1_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        nn4_2_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        self.conv4_1_l = GINConv(nn4_1_l, train_eps=True)
        self.conv4_2_l = GINConv(nn4_2_l, train_eps=True)
        self.conv4_1_g = GINConv(nn4_1_g, train_eps=True)
        self.conv4_2_g = GINConv(nn4_2_g, train_eps=True)
        self.mlp_4 = Sequential(Linear(4 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn5_1_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        nn5_2_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        nn5_1_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        nn5_2_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        self.conv5_1_l = GINConv(nn5_1_l, train_eps=True)
        self.conv5_2_l = GINConv(nn5_2_l, train_eps=True)
        self.conv5_1_g = GINConv(nn5_1_g, train_eps=True)
        self.conv5_2_g = GINConv(nn5_2_g, train_eps=True)
        self.mlp_5 = Sequential(Linear(4 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        nn6_1_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        nn6_2_l = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        nn6_1_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        nn6_2_g = Sequential(Linear(dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                             torch.nn.BatchNorm1d(dim), ReLU())
        self.conv6_1_l = GINConv(nn6_1_l, train_eps=True)
        self.conv6_2_l = GINConv(nn6_2_l, train_eps=True)
        self.conv6_1_g = GINConv(nn6_1_g, train_eps=True)
        self.conv6_2_g = GINConv(nn6_2_g, train_eps=True)
        self.mlp_6 = Sequential(Linear(4 * dim, dim), torch.nn.BatchNorm1d(dim), ReLU(), Linear(dim, dim),
                                torch.nn.BatchNorm1d(dim), ReLU())

        self.set2set = Set2Set(1 * dim, processing_steps=6)
        self.fc1 = Linear(2 * dim, dim)
        self.fc4 = Linear(dim, 12)

    def forward(self, data):
        x = data.x

        x_1 = F.relu(self.conv1_1_l(x, data.edge_index_1_l))
        x_2 = F.relu(self.conv1_2_l(x, data.edge_index_2_l))
        x_3 = F.relu(self.conv1_1_g(x, data.edge_index_1_g))
        x_4 = F.relu(self.conv1_2_g(x, data.edge_index_2_g))
        x_1_r = self.mlp_1(torch.cat([x_1, x_3, x_2, x_4], dim=-1))

        x_1 = F.relu(self.conv2_1_l(x_1_r, data.edge_index_1_l))
        x_2 = F.relu(self.conv2_2_l(x_1_r, data.edge_index_2_l))
        x_3 = F.relu(self.conv2_1_g(x_1_r, data.edge_index_1_g))
        x_4 = F.relu(self.conv2_2_g(x_1_r, data.edge_index_2_g))
        x_2_r = self.mlp_2(torch.cat([x_1, x_3, x_2, x_4], dim=-1))

        x_1 = F.relu(self.conv3_1_l(x_2_r, data.edge_index_1_l))
        x_2 = F.relu(self.conv3_2_l(x_2_r, data.edge_index_2_l))
        x_3 = F.relu(self.conv3_1_g(x_2_r, data.edge_index_1_g))
        x_4 = F.relu(self.conv3_2_g(x_2_r, data.edge_index_2_g))
        x_3_r = self.mlp_3(torch.cat([x_1, x_3, x_2, x_4], dim=-1))

        x_1 = F.relu(self.conv4_1_l(x_3_r, data.edge_index_1_l))
        x_2 = F.relu(self.conv4_2_l(x_3_r, data.edge_index_2_l))
        x_3 = F.relu(self.conv4_1_g(x_3_r, data.edge_index_1_g))
        x_4 = F.relu(self.conv4_2_g(x_3_r, data.edge_index_2_g))
        x_4_r = self.mlp_4(torch.cat([x_1, x_3, x_2, x_4], dim=-1))

        x_1 = F.relu(self.conv5_1_l(x_4_r, data.edge_index_1_l))
        x_2 = F.relu(self.conv5_2_l(x_4_r, data.edge_index_2_l))
        x_3 = F.relu(self.conv5_1_g(x_4_r, data.edge_index_1_g))
        x_4 = F.relu(self.conv5_2_g(x_4_r, data.edge_index_2_g))
        x_5_r = self.mlp_5(torch.cat([x_1, x_3, x_2, x_4], dim=-1))

        x_1 = F.relu(self.conv6_1_l(x_5_r, data.edge_index_1_l))
        x_2 = F.relu(self.conv6_2_l(x_5_r, data.edge_index_2_l))
        x_3 = F.relu(self.conv6_1_g(x_5_r, data.edge_index_1_g))
        x_4 = F.relu(self.conv6_2_g(x_5_r, data.edge_index_2_g))
        x_6_r = self.mlp_6(torch.cat([x_1, x_3, x_2, x_4], dim=-1))

        x = x_6_r

        x = self.set2set(x, data.batch)
        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'Alchemy')
dataset = Alchemy_malkin(path, transform=MyTransform())

mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean.to(device), std.to(device)

train_dataset = dataset[0:50000].shuffle()
val_dataset = dataset[50000:55000].shuffle()
test_dataset = dataset[55000:60000].shuffle()

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

        return error.mean().item(), error_log.mean().item()


    best_val_error = None
    test_error = None
    test_error_log = None
    for epoch in range(1, 501):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss, train_error = train()
        val_error, _ = test(val_loader)

        scheduler.step(val_error)
        if best_val_error is None or val_error <= best_val_error:
            test_error, test_error_log = test(test_loader)
            best_val_error = val_error

        print(epoch, lr, loss, val_error, test_error)

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
