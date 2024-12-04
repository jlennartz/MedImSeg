import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

RANDOM_SEED = 42
COMPONENTS_COUNT = 3 # number of RGB components

def mds_to_rgb(feature_map: np.ndarray):
    """
    Maps multi-channel feature maps to RGB using MDS.
    :param feature_map: Numpy array of shape (C, H, W)
    :return: RGB image of shape (H, W, 3)
    """
    C, H, W = feature_map.shape
    feature_map_flat = feature_map.reshape(C, -1).T # Shape: (H*W, C)
    print(feature_map_flat.shape)
    
    distance_matrix = pairwise_distances(feature_map_flat, metric='euclidean')
    embedding = MDS(n_components=COMPONENTS_COUNT, dissimilarity='precomputed', random_state=RANDOM_SEED)
    feature_transformed = embedding.fit_transform(distance_matrix)  # Shape: (H*W, 3)
    
    return feature_transformed.reshape(H, W, COMPONENTS_COUNT)

def pca_to_rgb(feature_map):
    C, H, W = feature_map.shape
    feature_map_flat = feature_map.reshape(C, -1).T
    
    pca = PCA(n_components=COMPONENTS_COUNT, random_state=RANDOM_SEED)
    feature_transformed = pca.fit_transform(feature_map_flat)
    
    return feature_transformed.reshape(H, W, 3)
