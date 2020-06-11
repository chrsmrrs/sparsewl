import sys

sys.path.insert(0, '..')
sys.path.insert(0, '.')

import auxiliarymethods.datasets as dp
import preprocessing as pre

import os.path as osp
import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool, GINConv

from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import pickle


class ZINC_wl_1(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(ZINC_wl_1, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "ZINC_train_1"

    @property
    def processed_file_names(self):
        return "ZINC_train_1"

    def download(self):
        pass

    def process(self):
        data_list = []

        indices_train = list(range(0, 220011))
        indices_val = list(range(0, 24445))
        indices_test = list(range(0, 5000))

        dp.get_dataset("ZINC_train", regression=True)
        dp.get_dataset("ZINC_test", regression=True)
        dp.get_dataset("ZINC_val", regression=True)
        node_labels = pre.get_all_node_labels_ZINC(True, True, indices_train, indices_val, indices_test)

        targets = pre.read_targets("ZINC_train", indices_train)
        targets.extend(pre.read_targets("ZINC_val", indices_val))
        targets.extend(pre.read_targets("ZINC_test", indices_test))

        node_labels_1 = node_labels[0:50000]
        matrices = pre.get_all_matrices_wl("ZINC_train", list(range(0, 50000)))
        targets_1 = targets[0:50000]
        for i, m in enumerate(matrices):
            edge_index_1 = torch.tensor(matrices[i][0]).t().contiguous()
            edge_index_2 = torch.tensor(matrices[i][1]).t().contiguous()

            data = Data()
            data.edge_index_1 = edge_index_1
            data.edge_index_2 = edge_index_2

            data.x = torch.from_numpy(np.array(node_labels_1[i])).to(torch.float)
            data.y = data.y = torch.from_numpy(np.array([targets_1[i]])).to(torch.float)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class ZINC_wl_2(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(ZINC_wl_2, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "ZINC_train_2"

    @property
    def processed_file_names(self):
        return "ZINC_train_2"

    def download(self):
        pass

    def process(self):
        data_list = []

        indices_train = list(range(0, 220011))
        indices_val = list(range(0, 24445))
        indices_test = list(range(0, 5000))

        dp.get_dataset("ZINC_train", regression=True)
        dp.get_dataset("ZINC_test", regression=True)
        dp.get_dataset("ZINC_val", regression=True)
        node_labels = pre.get_all_node_labels_ZINC(True, True, indices_train, indices_val, indices_test)

        targets = pre.read_targets("ZINC_train", indices_train)
        targets.extend(pre.read_targets("ZINC_val", indices_val))
        targets.extend(pre.read_targets("ZINC_test", indices_test))

        node_labels_1 = node_labels[50000:100000]
        matrices = pre.get_all_matrices_wl("ZINC_train", list(range(50000, 100000)))
        targets_1 = targets[50000:100000]
        for i, m in enumerate(matrices):
            edge_index_1 = torch.tensor(matrices[i][0]).t().contiguous()
            edge_index_2 = torch.tensor(matrices[i][1]).t().contiguous()

            data = Data()
            data.edge_index_1 = edge_index_1
            data.edge_index_2 = edge_index_2

            # one_hot = np.eye(492)[node_labels[i]]
            data.x = torch.from_numpy(np.array(node_labels_1[i])).to(torch.float)
            data.y = data.y = torch.from_numpy(np.array([targets_1[i]])).to(torch.float)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class ZINC_wl_3(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(ZINC_wl_3, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "ZINC_train_3"

    @property
    def processed_file_names(self):
        return "ZINC_train_3"

    def download(self):
        pass

    def process(self):
        data_list = []

        indices_train = list(range(0, 220011))
        indices_val = list(range(0, 24445))
        indices_test = list(range(0, 5000))

        dp.get_dataset("ZINC_train", regression=True)
        dp.get_dataset("ZINC_test", regression=True)
        dp.get_dataset("ZINC_val", regression=True)
        node_labels = pre.get_all_node_labels_ZINC(True, True, indices_train, indices_val, indices_test)

        targets = pre.read_targets("ZINC_train", indices_train)
        targets.extend(pre.read_targets("ZINC_val", indices_val))
        targets.extend(pre.read_targets("ZINC_test", indices_test))

        node_labels_1 = node_labels[100000:150000]
        matrices = pre.get_all_matrices_wl("ZINC_train", list(range(100000, 150000)))
        targets_1 = targets[100000:150000]
        for i, m in enumerate(matrices):
            edge_index_1 = torch.tensor(matrices[i][0]).t().contiguous()
            edge_index_2 = torch.tensor(matrices[i][1]).t().contiguous()

            data = Data()
            data.edge_index_1 = edge_index_1
            data.edge_index_2 = edge_index_2

            data.x = torch.from_numpy(np.array(node_labels_1[i])).to(torch.float)
            data.y = data.y = torch.from_numpy(np.array([targets_1[i]])).to(torch.float)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class ZINC_wl_4(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(ZINC_wl_4, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "ZINC_train_4"

    @property
    def processed_file_names(self):
        return "ZINC_train_4"

    def download(self):
        pass

    def process(self):
        data_list = []

        indices_train = list(range(0, 220011))
        indices_val = list(range(0, 24445))
        indices_test = list(range(0, 5000))

        dp.get_dataset("ZINC_train", regression=True)
        dp.get_dataset("ZINC_test", regression=True)
        dp.get_dataset("ZINC_val", regression=True)
        node_labels = pre.get_all_node_labels_ZINC(True, True, indices_train, indices_val, indices_test)

        targets = pre.read_targets("ZINC_train", indices_train)
        targets.extend(pre.read_targets("ZINC_val", indices_val))
        targets.extend(pre.read_targets("ZINC_test", indices_test))

        node_labels_1 = node_labels[150000:200000]
        matrices = pre.get_all_matrices_wl("ZINC_train", list(range(150000, 200000)))
        targets_1 = targets[150000:200000]
        for i, m in enumerate(matrices):
            edge_index_1 = torch.tensor(matrices[i][0]).t().contiguous()
            edge_index_2 = torch.tensor(matrices[i][1]).t().contiguous()

            data = Data()
            data.edge_index_1 = edge_index_1
            data.edge_index_2 = edge_index_2

            # one_hot = np.eye(492)[node_labels[i]]
            data.x = torch.from_numpy(np.array(node_labels_1[i])).to(torch.float)
            data.y = data.y = torch.from_numpy(np.array([targets_1[i]])).to(torch.float)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class ZINC_wl_5(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(ZINC_wl_5, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "ZINC_train_5"

    @property
    def processed_file_names(self):
        return "ZINC_train_5"

    def download(self):
        pass

    def process(self):
        data_list = []

        indices_train = list(range(0, 220011))
        indices_val = list(range(0, 24445))
        indices_test = list(range(0, 5000))

        dp.get_dataset("ZINC_train", regression=True)
        dp.get_dataset("ZINC_test", regression=True)
        dp.get_dataset("ZINC_val", regression=True)
        node_labels = pre.get_all_node_labels_ZINC(True, True, indices_train, indices_val, indices_test)

        targets = pre.read_targets("ZINC_train", indices_train)
        targets.extend(pre.read_targets("ZINC_val", indices_val))
        targets.extend(pre.read_targets("ZINC_test", indices_test))

        node_labels_1 = node_labels[200000:220011]
        matrices = pre.get_all_matrices_wl("ZINC_train", list(range(200000, 220011)))
        targets_1 = targets[200000:220011]
        for i, m in enumerate(matrices):
            edge_index_1 = torch.tensor(matrices[i][0]).t().contiguous()
            edge_index_2 = torch.tensor(matrices[i][1]).t().contiguous()

            data = Data()
            data.edge_index_1 = edge_index_1
            data.edge_index_2 = edge_index_2

            data.x = torch.from_numpy(np.array(node_labels_1[i])).to(torch.float)
            data.y = data.y = torch.from_numpy(np.array([targets_1[i]])).to(torch.float)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class ZINC_wl_all(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(ZINC_wl_all, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "ZINC_train_all"

    @property
    def processed_file_names(self):
        return "ZINC_train_all"

    def download(self):
        pass

    def process(self):
        path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'ZINC')
        dataset_1 = ZINC_wl_1(path, transform=MyTransform()).shuffle()
        path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'ZINC')
        dataset_2 = ZINC_wl_2(path, transform=MyTransform()).shuffle()
        path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'ZINC')
        dataset_3 = ZINC_wl_3(path, transform=MyTransform()).shuffle()
        path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'ZINC')
        dataset_4 = ZINC_wl_4(path, transform=MyTransform()).shuffle()
        path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'ZINC')
        dataset_5 = ZINC_wl_5(path, transform=MyTransform()).shuffle()

        dataset = torch.utils.data.ConcatDataset([dataset_1, dataset_2, dataset_3, dataset_4, dataset_5])
        data_list = []

        for data in dataset:
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class ZINC_wl_val(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(ZINC_wl_val, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "ZINC_wl_vadlfd"

    @property
    def processed_file_names(self):
        return "ZINC_wl_valdfd"

    def download(self):
        pass

    def process(self):
        data_list = []

        dp.get_dataset("ZINC_val", regression=True)
        node_labels = pre.get_all_node_labels("ZINC_full", True, True)
        targets = pre.read_targets("ZINC_val", list(range(0, 24445)))

        node_labels_1 = node_labels[225011:249456]
        matrices = pre.get_all_matrices_wl("ZINC_val", list(range(0, 24445)))
        targets_1 = targets
        for i, m in enumerate(matrices):
            edge_index_1 = torch.tensor(matrices[i][0]).t().contiguous()
            edge_index_2 = torch.tensor(matrices[i][1]).t().contiguous()

            data = Data()
            data.edge_index_1 = edge_index_1
            data.edge_index_2 = edge_index_2

            data.x = torch.from_numpy(np.array(node_labels_1[i])).to(torch.float)
            data.y = data.y = torch.from_numpy(np.array([targets_1[i]])).to(torch.float)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class ZINC_wl_test(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(ZINC_wl_test, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "ZINC_wl_testfd"

    @property
    def processed_file_names(self):
        return "ZINC_wl_testfd"

    def download(self):
        pass

    def process(self):
        data_list = []

        dp.get_dataset("ZINC_test", regression=True)

        node_labels = pre.get_all_node_labels("ZINC_full", True, True)
        targets = pre.read_targets("ZINC_test", list(range(0, 5000)))

        node_labels_1 = node_labels[220011:225011]
        matrices = pre.get_all_matrices_wl("ZINC_test", list(range(0, 5000)))
        targets_1 = targets
        for i, m in enumerate(matrices):
            edge_index_1 = torch.tensor(matrices[i][0]).t().contiguous()
            edge_index_2 = torch.tensor(matrices[i][1]).t().contiguous()

            data = Data()
            data.edge_index_1 = edge_index_1
            data.edge_index_2 = edge_index_2

            data.x = torch.from_numpy(np.array(node_labels_1[i])).to(torch.float)
            data.y = data.y = torch.from_numpy(np.array([targets_1[i]])).to(torch.float)

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

        num_features = 652

        nn1_1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        nn1_2 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1_1 = GINConv(nn1_1, train_eps=True)
        self.conv1_2 = GINConv(nn1_2, train_eps=True)
        self.bn1 = torch.nn.BatchNorm1d(dim)
        self.mlp_1 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        nn2_1 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        nn2_2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2_1 = GINConv(nn2_1, train_eps=True)
        self.conv2_2 = GINConv(nn2_2, train_eps=True)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        self.mlp_2 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        nn3_1 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        nn3_2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3_1 = GINConv(nn3_1, train_eps=True)
        self.conv3_2 = GINConv(nn3_2, train_eps=True)
        self.bn3 = torch.nn.BatchNorm1d(dim)
        self.mlp_3 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        nn4_1 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        nn4_2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4_1 = GINConv(nn4_1, train_eps=True)
        self.conv4_2 = GINConv(nn4_2, train_eps=True)
        self.bn4 = torch.nn.BatchNorm1d(dim)
        self.mlp_4 = Sequential(Linear(2 * dim, dim), ReLU(), Linear(dim, dim))

        self.fc1 = Linear(4 * dim, dim)
        self.fc2 = Linear(dim, dim)
        self.fc3 = Linear(dim, dim)
        self.fc4 = Linear(dim, 1)

    def forward(self, data):
        x = data.x
        x = x.long()

        x_new = torch.zeros(x.size(0), 652).to(device)
        x_new[range(x_new.shape[0]), x.view(1, x.size(0))] = 1

        x_1 = F.relu(self.conv1_1(x_new, data.edge_index_1))
        x_2 = F.relu(self.conv1_2(x_new, data.edge_index_2))
        x_1_r = self.mlp_1(torch.cat([x_1, x_2], dim=-1))
        x_1_r = self.bn1(x_1_r)

        x_1 = F.relu(self.conv2_1(x_1_r, data.edge_index_1))
        x_2 = F.relu(self.conv2_2(x_1_r, data.edge_index_2))
        x_2_r = self.mlp_2(torch.cat([x_1, x_2], dim=-1))
        x_2_r = self.bn2(x_2_r)

        x_1 = F.relu(self.conv3_1(x_2_r, data.edge_index_1))
        x_2 = F.relu(self.conv3_2(x_2_r, data.edge_index_2))
        x_3_r = self.mlp_3(torch.cat([x_1, x_2], dim=-1))
        x_3_r = self.bn3(x_3_r)

        x_1 = F.relu(self.conv4_1(x_3_r, data.edge_index_1))
        x_2 = F.relu(self.conv4_2(x_3_r, data.edge_index_2))
        x_4_r = self.mlp_4(torch.cat([x_1, x_2], dim=-1))
        x_4_r = self.bn4(x_4_r)

        x = torch.cat([x_1_r, x_2_r, x_3_r, x_4_r], dim=-1)
        x = global_mean_pool(x, data.batch)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x.view(-1)


plot_all = []
results = []

for _ in range(5):
    plot_it = []
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'ZINC')
    dataset_6 = ZINC_wl_val(path, transform=MyTransform()).shuffle()
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'ZINC')
    dataset_7 = ZINC_wl_test(path, transform=MyTransform()).shuffle()

    train = ZINC_wl_all(path, transform=MyTransform()).shuffle()

    batch_size = 25
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_6, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_7, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NetGIN(256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=5,
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
            loss_all += loss.item() * data.num_graphs
            optimizer.step()

        return loss_all / (len(train_loader.dataset))


    def test(loader):
        model.eval()
        error = 0

        for data in loader:
            data = data.to(device)
            error += (model(data) - data.y).abs().sum().item()  # MAE
        return error / len(loader.dataset)

    best_val_error = None
    test_error = None
    for epoch in range(1, 1001):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train()
        val_error = test(val_loader)
        scheduler.step(val_error)

        if best_val_error is None or val_error <= best_val_error:
            test_error = test(test_loader)
            best_val_error = val_error

        plot_it.append([loss, val_error, test_error])
        print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
              'Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error))

        if lr < 0.000001:
            print("Converged.")
            plot_all.append(plot_it)
            break

print("########################")
print(results)
results = np.array(results)
print(results.mean(), results.std())
with open('plot_ZINC_wl_all', 'wb') as fp:
    pickle.dump(plot_all, fp)
