import torch
from torch.utils.data import Dataset as image_dataset
from torch_geometric.data import Dataset as graph_dataset
from torch_geometric.data import Data
import os
import numpy as np
import utils


class Brain_image(image_dataset):
    def __init__(self, data_path, modality):
        self.data_path = data_path
        self.modality = modality
        self.image_list, self.label_list, self.name_list = self.get_data()

    def __getitem__(self, index):
        # get item by index
        image, label, name = np.load(self.image_list[index]), np.load(self.label_list[index]), self.name_list[index]

        # transform numpy to tensor
        image = torch.from_numpy(image)

        # add channel dimension for image
        image = torch.unsqueeze(image, dim=0)

        return image, label, name

    def __len__(self):
        return len(self.image_list)

    def get_data(self):
        image_list = list()
        label_list = list()
        name_list = list()

        # define file paths
        image_path = os.path.join(self.data_path, str(self.modality))
        label_path = os.path.join(self.data_path, 'Age')

        sub_dir = os.listdir(image_path)
        sub_dir.sort(key=lambda x: int(x[:-4]))

        # load data and label
        for name in sub_dir:
            image = os.path.join(image_path, name)
            label = os.path.join(label_path, name)
            image_list.append(image)
            label_list.append(label)
            name_list.append(name)

        return image_list, label_list, name_list


class Brain_network(graph_dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        # define file paths
        node_path = os.path.join(self.raw_dir, 'Fusion/node_feature')
        adjacency_path = os.path.join(self.raw_dir, 'Fusion/adjacency_matrix')
        age_path = os.path.join(self.raw_dir, 'Fusion/age')

        sub_dir = os.listdir(node_path)
        sub_dir.sort(key=lambda x: int(x[:-4]))

        # load data and label
        index = 0
        for name in sub_dir:
            node_feature = np.load(os.path.join(node_path, name))
            adjacency_matrix = np.load(os.path.join(adjacency_path, name))
            age = np.load(os.path.join(age_path, name))

            # get edges from adjacency matrix
            edge_index = utils.get_edge(adjacency_matrix)

            # transform numpy to tensor
            edge_index = torch.from_numpy(edge_index)
            node_feature = torch.from_numpy(node_feature)
            label = torch.from_numpy(age)

            # obtain graph data
            graph_data = Data(x=node_feature, y=label, edge_index=edge_index)

            # graph data preprocessing
            if self.pre_filter is not None and not self.pre_filter(graph_data):
                continue

            if self.pre_transform is not None:
                graph_data = self.pre_transform(graph_data)

            # save graph data
            torch.save(graph_data, os.path.join(self.processed_dir, f'data_{index}.pt'))
            index += 1

    def len(self):
        return len(os.listdir(self.processed_dir)) - 2

    def get(self, index):
        # get graph data by index
        graph_data = torch.load(os.path.join(self.processed_dir, f'data_{index}.pt'))
        return graph_data
