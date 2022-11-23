import numpy as np
import heapq
from sklearn.metrics.pairwise import cosine_similarity


def get_roi_feature(feature_map_s, feature_map_m, template_s, template_m):
    roi_feature = list()
    # special ROIs are small brain regions that may be ignored on the downsampled template
    special_ROI = [21, 22, 35, 36, 41, 42, 75, 76, 79, 80]

    # 90 ROIs
    for i in range(1, 91):
        # feature aggregation at different scales
        if i not in special_ROI:
            roi_template = (template_s == i).astype(np.uint8)[np.newaxis, :, :, :]
            feature = np.sum(roi_template * feature_map_s, axis=(1, 2, 3)) / np.sum(roi_template)
            roi_feature.append(feature)
        else:
            roi_template = (template_m == i).astype(np.uint8)[np.newaxis, :, :, :]
            feature = np.sum(roi_template * feature_map_m, axis=(1, 2, 3)) / np.sum(roi_template)
            roi_feature.append(np.concatenate((feature, feature), axis=0))

    roi_feature = np.array(roi_feature)
    return roi_feature


def get_adjacency_matrix(roi_feature, distance_matrix, k_num):
    # calculate feature similarity as feature matrix
    feature_matrix = cosine_similarity(roi_feature)

    # get sparse and binary matrix
    feature_matrix = get_binary_matrix(feature_matrix, k_num)
    distance_matrix = get_binary_matrix(-1*distance_matrix, k_num)

    # combine feature and distance matrix
    adjacency_matrix = combine_connection_matrix(distance_matrix, feature_matrix)

    return adjacency_matrix


def get_binary_matrix(connection_matrix, k_num):
    roi_num = connection_matrix.shape[0]

    # get sparse and binary connections
    for i in range(roi_num):
        node_connection = connection_matrix[i, :]
        # choose the k closest positions but exclude the node itself
        position = heapq.nlargest(k_num + 1, range(len(node_connection)), node_connection.__getitem__)
        sparse_connection = np.zeros(roi_num, dtype=np.uint8)
        for j in range(k_num + 1):
            sparse_connection[position[j]] = 1
        sparse_connection[i] = 0
        connection_matrix[i, :] = sparse_connection

    # complete connection matrix
    for i in range(roi_num):
        for j in range(roi_num):
            if connection_matrix[i, j] == 1:
                connection_matrix[j, i] = 1

    return connection_matrix


def combine_connection_matrix(distance_matrix, feature_matrix):
    roi_num = distance_matrix.shape[0]

    # combine connections of two matrices
    for i in range(roi_num):
        for j in range(roi_num):
            if feature_matrix[i, j] == 1:
                distance_matrix[i, j] = 1

    return distance_matrix


def combine_modality_matrix(mri_matrix, dti_matrix, k_num):
    # initialize fusion matrix
    roi_num = mri_matrix.shape[0]
    fusion_matrix = np.zeros((roi_num*2, roi_num*2), dtype=np.uint8)
    fusion_matrix[0:roi_num, 0:roi_num] = mri_matrix
    fusion_matrix[roi_num:2*roi_num, roi_num:2*roi_num] = dti_matrix

    # calculate similarity as modality connections
    connection_matrix = cosine_similarity(mri_matrix, dti_matrix)

    # get sparse and binary connections
    for i in range(roi_num):
        node_connection = connection_matrix[i, :]
        # choose the k closest positions
        position = heapq.nlargest(k_num, range(len(node_connection)), node_connection.__getitem__)
        sparse_connection = np.zeros(roi_num, dtype=np.uint8)
        for j in range(k_num):
            sparse_connection[position[j]] = 1
        connection_matrix[i, :] = sparse_connection

    # complete fusion matrix
    fusion_matrix[0:roi_num, roi_num:2*roi_num] = connection_matrix
    for i in range(roi_num):
        for j in range(roi_num, 2*roi_num):
            if fusion_matrix[i, j] == 1:
                fusion_matrix[j, i] = 1

    return fusion_matrix


def get_edge(adjacency_matrix):
    edge = list()
    roi_num = adjacency_matrix.shape[0]

    # save edge by [Source Node, Target Node] from adjacency matrix
    for i in range(roi_num):
        for j in range(roi_num):
            if adjacency_matrix[i, j] == 1:
                edge.append(np.array([i, j]))

    # transpose edge list for graph construction
    edge = np.swapaxes(np.array(edge), axis1=0, axis2=1)

    return edge
