import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import torch
from torch.utils.data import DataLoader
import Data
import utils
import ConvNet
import logging
import numpy as np
import shutil


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def BrainNetwork_single_modal(modality):
    # experiment settings
    model_path = './trained/CAE'
    data_path = './data'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data preparation
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    dataLoader = DataLoader(Data.Brain_image(data_path, modality), batch_size=1, shuffle=False, **kwargs)

    # network construction
    net = ConvNet.Feature_Extraction(nChannels=16)

    # move the network to GPU/CPU
    net = torch.nn.DataParallel(net)
    net = net.to(device)

    # get trained model
    save_model = torch.load(os.path.join(model_path, 'model_' + modality + '_CAE.pth'))
    model_dict = net.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    net.load_state_dict(model_dict)

    # load ROI template
    template_s = np.load('./template/aal90_template_20x24x20.npy')
    template_m = np.load('./template/aal90_template_40x48x40.npy')

    # define file paths
    node_path = './brain_network/raw/' + modality + '/node_feature'
    adjacency_path = node_path.replace('node_feature', 'adjacency_matrix')
    age_path = node_path.replace('node_feature', 'age')

    # reset folders
    shutil.rmtree(node_path)
    os.mkdir(node_path)
    shutil.rmtree(adjacency_path)
    os.mkdir(adjacency_path)
    shutil.rmtree(age_path)
    os.mkdir(age_path)

    # brain network construction
    net.eval()
    for batch_idx, (image, label, name) in enumerate(dataLoader):
        image = image.to(device)

        # obtain feature maps from network
        with torch.no_grad():
            feature_map_s, feature_map_m = net(image)

        feature_map_s = feature_map_s.cpu().detach().numpy().squeeze()
        feature_map_m = feature_map_m.cpu().detach().numpy().squeeze()

        # get ROI feature as node features
        roi_feature = utils.get_roi_feature(feature_map_s, feature_map_m, template_s, template_m)
        np.save(os.path.join(node_path, name[0]), roi_feature)

        # get adjacency matrix
        distance_matrix = np.load('./template/aal90_distance_matrix.npy')
        adjacency_matrix = utils.get_adjacency_matrix(roi_feature, distance_matrix, k_num=8)
        np.save(os.path.join(adjacency_path, name[0]), adjacency_matrix)

        # get subject age
        np.save(os.path.join(age_path, name[0]), label[0])

    logging.info('Brain network construction of {} modality is completed.'.format(modality))


def BrainNetwork_multi_modal():
    # define file paths
    mri_path = './brain_network/raw/MRI'
    dti_path = './brain_network/raw/DTI'
    fusion_path = './brain_network/raw/Fusion'

    mri_node_path = os.path.join(mri_path, 'node_feature')
    dti_node_path = os.path.join(dti_path, 'node_feature')
    fusion_node_path = os.path.join(fusion_path, 'node_feature')

    mri_adjacency_path = os.path.join(mri_path, 'adjacency_matrix')
    dti_adjacency_path = os.path.join(dti_path, 'adjacency_matrix')
    fusion_adjacency_path = os.path.join(fusion_path, 'adjacency_matrix')

    mri_age_path = os.path.join(mri_path, 'age')
    dti_age_path = os.path.join(dti_path, 'age')
    fusion_age_path = os.path.join(fusion_path, 'age')

    # reset folders
    shutil.rmtree(fusion_node_path)
    os.mkdir(fusion_node_path)
    shutil.rmtree(fusion_adjacency_path)
    os.mkdir(fusion_adjacency_path)
    shutil.rmtree(fusion_age_path)
    os.mkdir(fusion_age_path)

    # combine node features
    sub_dir = os.listdir(mri_node_path)
    for name in sub_dir:
        mri_node_feature = np.load(os.path.join(mri_node_path, name))
        dti_node_feature = np.load(os.path.join(dti_node_path, name))
        fusion_node_feature = np.concatenate((mri_node_feature, dti_node_feature), axis=0)
        np.save(os.path.join(fusion_node_path, name), fusion_node_feature)

    # combine adjacency matrix
    sub_dir = os.listdir(mri_adjacency_path)
    for name in sub_dir:
        mri_adj_matrix = np.load(os.path.join(mri_adjacency_path, name))
        dti_adj_matrix = np.load(os.path.join(dti_adjacency_path, name))
        fusion_adj_matrix = utils.combine_modality_matrix(mri_adj_matrix, dti_adj_matrix, k_num=8)
        np.save(os.path.join(fusion_adjacency_path, name), fusion_adj_matrix)

    # get subject age
    sub_dir = os.listdir(mri_age_path)
    for name in sub_dir:
        shutil.copy(os.path.join(dti_age_path, name), os.path.join(fusion_age_path, name))

    logging.info('Multimodal brain network construction is completed.')


if __name__ == '__main__':
    # brain network construction for MRI
    BrainNetwork_single_modal(modality='MRI')

    # brain network construction for DTI
    BrainNetwork_single_modal(modality='DTI')

    # multimodal brain network construction
    BrainNetwork_multi_modal()
