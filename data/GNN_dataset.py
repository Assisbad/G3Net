import os
import numpy as np
import torch
from PIL import Image
import networkx as nx
from data.pix2pix_dataset import Pix2pixDataset
from data.base_dataset import get_params, get_transform


class ImageGraph:
    def __init__(self, base_path):
        self.base_path = base_path

    def load_images_and_masks(self, images_path, combined_masks_path):
        images = []
        masks = []


        image_dict = {}
        for file_name in sorted(os.listdir(images_path)):
            if file_name.startswith('slice') and file_name.endswith('.png'):
                image_path = os.path.join(images_path, file_name)
                with Image.open(image_path) as image:
                    image_dict[file_name] = image.copy()


        mask_dict = {file_name: None for file_name in image_dict.keys()}


        for file_name in sorted(os.listdir(combined_masks_path)):
            if file_name.startswith('slice') and file_name.endswith('.png'):
                mask_index = file_name.split('_')[-1].split('.')[0]
                corresponding_image_name = f'slice_{mask_index}.png'
                mask_path = os.path.join(combined_masks_path, file_name)
                if os.path.exists(mask_path):
                    with Image.open(mask_path).convert('L') as mask:
                        mask_dict[corresponding_image_name] = mask.copy()


        for file_name in sorted(image_dict.keys()):
            image = image_dict[file_name]
            mask = mask_dict[file_name]

            if mask is None:
                continue

            images.append((file_name, image))
            masks.append((file_name, mask))

        return images, masks

    def process_all(self):
        all_graphs = {}
        skipped_folders = 0

        for patient_folder in os.listdir(self.base_path):
            patient_path = os.path.join(self.base_path, patient_folder)
            if os.path.isdir(patient_path):
                images_path = os.path.join(patient_path, 'image')
                combined_masks_path = os.path.join(patient_path, 'label')
                if os.path.isdir(images_path) and os.path.isdir(combined_masks_path):
                    images, masks = self.load_images_and_masks(images_path, combined_masks_path)
                    if images:
                        graph = self.create_graph(images, masks, patient_folder)
                        all_graphs[(patient_folder)] = graph
                    else:
                        skipped_folders += 1

        return all_graphs

    def create_graph(self, images, masks, patient_folder):
        G = nx.Graph()
        num_nodes = len(images)


        for i in range(num_nodes):
            image_name, image = images[i]
            mask_name, mask = masks[i]

            if mask is not None:
                G.add_node(i, image=image, mask=mask, folder_path=patient_folder)


        valid_nodes = list(G.nodes)
        for i in range(len(valid_nodes) - 1):
            G.add_edge(valid_nodes[i], valid_nodes[i + 1])


        for i in valid_nodes:
            G.add_edge(i, i)


        if valid_nodes:
            global_node = max(valid_nodes) + 1
            G.add_node(global_node, image="global_node", mask=None)
            for i in valid_nodes:
                G.add_edge(i, global_node)


            G.add_edge(global_node, global_node)

        return G

    def adjacency_matrix(self, graph):
        return nx.adjacency_matrix(graph).todense()


class GNNDataset(Pix2pixDataset):
    def initialize(self, opt):
        self.opt = opt
        self.base_path = opt.dataroot
        self.image_graph = ImageGraph(self.base_path)
        self.all_graphs = self.image_graph.process_all()

        # 默认按组处理图数据
        self.all_groups = []
        self.all_adjs = []
        for key, graph in self.all_graphs.items():
            group = []
            nodes_in_group = []
            for node, data in graph.nodes(data=True):
                if data['image'] != "global_node":
                    group.append({
                        'image': data['image'],
                        'mask': data['mask'],
                        'path': data['folder_path']
                    })
                    nodes_in_group.append(node)
            if group:
                self.all_groups.append(group)

                subgraph = graph.subgraph(nodes_in_group)
                adj_matrix = nx.adjacency_matrix(subgraph).todense()
                adj_tensor = torch.from_numpy(np.array(adj_matrix)).float()
                self.all_adjs.append(adj_tensor)
        self.dataset_size = len(self.all_groups)

    def get_paths(self, opt):
        return [], [], []

    def __getitem__(self, index):

        group = self.all_groups[index]
        adj = self.all_adjs[index]
        numnodes = len(group)
        images = []
        labels = []
        paths = []

        for sample in group:
            image = sample['image']
            mask = sample['mask']
            path = sample['path']


            params = get_params(self.opt, image.size)
            transform_image = get_transform(self.opt, params)
            image_tensor = transform_image(image.convert('L'))
            mask_np = np.array(mask)

            label_map = mask_np.astype(np.int64)
            label_tensor = torch.from_numpy(label_map).unsqueeze(0).long()

            images.append(image_tensor)
            labels.append(label_tensor)
            paths.append(path)


        images_tensor = torch.stack(images)
        labels_tensor = torch.stack(labels)

        inst_tensor = torch.zeros_like(labels_tensor)
        feat_tensor = torch.zeros_like(labels_tensor)

        return {
            'label': labels_tensor,
            'image': images_tensor,
            'instance': inst_tensor,
            'feat': feat_tensor,
            'path': paths,
            'adj': adj,
            'numnodes': numnodes
        }

    def __len__(self):
        return self.dataset_size

