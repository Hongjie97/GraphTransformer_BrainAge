import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import torch
from torch.utils.data import DataLoader
import Data
import utils
import ConvNet
import numpy as np
import shutil


def BrainNetwork_single_modal(modality):
    # some experiment settings
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

    # load ROIs template
    template1 = np.load('./template/aal116_template_20.npy')
    template2 = np.load('./template/aal116_template_40.npy')

    # define the file path
    node_path = './brain_network/raw/' + modality + '/node_feature'
    adjacency_path = node_path.replace('node_feature', 'adjacency_matrix')
    age_path = node_path.replace('node_feature', 'age')

    # reset the folders
    shutil.rmtree(node_path)
    os.mkdir(node_path)
    shutil.rmtree(adjacency_path)
    os.mkdir(adjacency_path)
    shutil.rmtree(age_path)
    os.mkdir(age_path)

    # brain network construction
    net.eval()

    # for sample data
    for batch_idx, (image, label, name) in enumerate(dataLoader):
        # obtain feature maps from network
        image = image.to(device)

        with torch.no_grad():
            feature_map1, feature_map2 = net(image)
        feature_map1 = feature_map1.cpu().detach().numpy().squeeze()
        feature_map2 = feature_map2.cpu().detach().numpy().squeeze()

        # get ROI feature
        roi_feature = utils.get_roi_feature(feature_map1, feature_map2, template1, template2)
        np.save(os.path.join(node_path, name[0]), roi_feature)

        # get adjacency matrix
        distance_matrix = np.load('./template/distance_aal116.npy')
        adjacency_matrix = utils.get_adjacency_matrix(roi_feature, distance_matrix, k_num=8)
        np.save(os.path.join(adjacency_path, name[0]), adjacency_matrix)

        # get age as label
        np.save(os.path.join(age_path, name[0]), label[0])


def BrainNetwork_multi_modal():
    # define the file path
    mri_path = './brain_network/raw/MRI'
    dti_path = './brain_network/raw/DTI'
    fusion_path = './brain_network/raw/Fusion'

    mri_path_node = os.path.join(mri_path, 'node_feature')
    dti_path_node = os.path.join(dti_path, 'node_feature')
    fusion_path_node = os.path.join(fusion_path, 'node_feature')

    mri_path_adjacency = os.path.join(mri_path, 'adjacency_matrix')
    dti_path_adjacency = os.path.join(dti_path, 'adjacency_matrix')
    fusion_path_adjacency = os.path.join(fusion_path, 'adjacency_matrix')

    mri_path_age = os.path.join(mri_path, 'age')
    dti_path_age = os.path.join(dti_path, 'age')
    fusion_path_age = os.path.join(fusion_path, 'age')

    # reset folders
    shutil.rmtree(os.path.join(fusion_path, 'node_feature'))
    os.mkdir(os.path.join(fusion_path, 'node_feature'))
    shutil.rmtree(os.path.join(fusion_path, 'adjacency_matrix'))
    os.mkdir(os.path.join(fusion_path, 'adjacency_matrix'))
    shutil.rmtree(os.path.join(fusion_path, 'age'))
    os.mkdir(os.path.join(fusion_path, 'age'))

    # combine node features
    sub_dir = os.listdir(mri_path_node)
    for name in sub_dir:
        mri_node_feature = np.load(os.path.join(mri_path_node, name))
        dti_node_feature = np.load(os.path.join(dti_path_node, name))
        fusion_node_feature = np.concatenate((mri_node_feature, dti_node_feature), axis=0)
        np.save(os.path.join(fusion_path_node, name), fusion_node_feature)

    # calculate adjacency matrix
    sub_dir = os.listdir(mri_path_adjacency)

    for name in sub_dir:
        mri_adj_matrix = np.load(os.path.join(mri_path_adjacency, name))
        dti_adj_matrix = np.load(os.path.join(dti_path_adjacency, name))
        fusion_adj_matrix = utils.combine_matrix_modality(mri_adj_matrix, dti_adj_matrix, k_num=8)
        np.save(os.path.join(fusion_path_adjacency, name), fusion_adj_matrix)

    # get subject age
    sub_dir = os.listdir(mri_path_age)

    for name in sub_dir:
        shutil.copy(os.path.join(dti_path_age, name), os.path.join(fusion_path_age, name))


if __name__ == '__main__':
    # brain network construction for MRI
    BrainNetwork_single_modal(modality='MRI')

    # brain network construction for DTI
    BrainNetwork_single_modal(modality='DTI')

    # multi-modality brain network construction
    BrainNetwork_multi_modal()
