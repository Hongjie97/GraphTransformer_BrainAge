import numpy as np
import heapq
from sklearn.metrics.pairwise import cosine_similarity


def get_roi_feature(feature_map1, feature_map2, template1, template2):
    roi_feature = list()
    special_ROI = [21, 22, 35, 36, 41, 42, 75, 76, 79, 80]

    # 90 ROIs
    for i in range(1, 91):
        if i not in special_ROI:
            roi_template = (template1 == i).astype(np.uint8)[np.newaxis, :, :, :]
            roi_feature.append(np.sum(roi_template * feature_map1, axis=(1, 2, 3)) / np.sum(roi_template))
        else:
            roi_template = (template2 == i).astype(np.uint8)[np.newaxis, :, :, :]
            features = np.sum(roi_template * feature_map2, axis=(1, 2, 3)) / np.sum(roi_template)
            new_features = np.concatenate((features, features), axis=0)
            roi_feature.append(new_features)

    roi_feature = np.array(roi_feature)
    return roi_feature


def get_adjacency_matrix(roi_feature, distance_matrix, k_num):
    # calculate feature similarity
    feature_matrix = cosine_similarity(roi_feature)

    # Sparse and binary the matrix
    feature_matrix = binary_matrix(feature_matrix, k=k_num)
    distance_matrix = binary_matrix(distance_matrix * -1, k=k_num)

    # combine feature and distance matrix
    adjacency_matrix = combine_matrix_connection(distance_matrix, feature_matrix)
    return adjacency_matrix


def binary_matrix(adjacency_matrix, k):
    roi_num = adjacency_matrix.shape[0]

    # TopK strategy for selecting the first K nodes
    for i in range(roi_num):
        single_node = adjacency_matrix[i, :]
        position = heapq.nlargest(k + 1, range(len(single_node)), single_node.__getitem__)
        binary_link = np.zeros(roi_num, dtype=np.uint8)
        for j in range(k + 1):
            binary_link[position[j]] = 1
        binary_link[i] = 0
        adjacency_matrix[i, :] = binary_link

    # directed graph to undirected graph
    for i in range(roi_num):
        for j in range(roi_num):
            if adjacency_matrix[i, j] == 1:
                adjacency_matrix[j, i] = 1

    return adjacency_matrix


def combine_matrix_connection(distance_matrix, feature_matrix):
    roi_num = distance_matrix.shape[0]

    # combine two matrix connections
    for i in range(roi_num):
        for j in range(roi_num):
            if feature_matrix[i, j] == 1:
                distance_matrix[i, j] = 1
    return distance_matrix


def combine_matrix_modality(mri_matrix, dti_matrix, k_num):
    # initialize the fusion matrix
    roi_num = mri_matrix.shape[0]
    fusion_matrix = np.zeros((roi_num * 2, roi_num * 2), dtype=np.uint8)
    fusion_matrix[0:roi_num, 0:roi_num] = mri_matrix
    fusion_matrix[roi_num:2*roi_num, roi_num:2*roi_num] = dti_matrix

    # calculate the correlations between connections of two modality
    connection_matrix = cosine_similarity(mri_matrix, dti_matrix)

    # binary the link between two modality
    k = k_num
    for i in range(roi_num):
        single_node = connection_matrix[i, :]
        position = heapq.nlargest(k, range(len(single_node)), single_node.__getitem__)
        binary_link = np.zeros(roi_num, dtype=np.uint8)
        for j in range(k):
            binary_link[position[j]] = 1
        connection_matrix[i, :] = binary_link

    # get the final fusion matrix
    fusion_matrix[0:roi_num, roi_num:2*roi_num] = connection_matrix

    for i in range(roi_num):
        for j in range(roi_num, 2*roi_num):
            if fusion_matrix[i, j] == 1:
                fusion_matrix[j, i] = 1

    return fusion_matrix


def get_edge(adjacency_matrix):
    edge_link = list()
    roi_num = adjacency_matrix.shape[0]

    # save edge by [Source Node, Target Node] from adjacency matrix
    for i in range(roi_num):
        for j in range(roi_num):
            if adjacency_matrix[i, j] == 1:
                edge_link.append(np.array([i, j]))

    edge_link = np.swapaxes(np.array(edge_link), axis1=0, axis2=1)
    return edge_link
