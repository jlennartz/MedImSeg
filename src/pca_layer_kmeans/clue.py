import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, DistributedSampler

from utils import SamplingStrategy, ActualSequentialSampler

class CLUESampling(SamplingStrategy):
    """
    Implements CLUE: Clustering via Uncertainty-weighted Embeddings for segmentation tasks.
    """
    def __init__(self, dset, train_idx, model, device, args, batch_size, balanced=False):
        super(CLUESampling, self).__init__(dset, train_idx, model, device, args)
        self.dset = dset
        self.train_idx = train_idx
        self.model = model
        self.device = device
        self.args = args
        self.batch_size = batch_size  # Explicitly use batch_size as a parameter
        self.cluster_type = args.cluster_type
        self.T = args.clue_softmax_t

        # [NEW] Retrieve new parameters from args
        self.use_uncertainty = args.use_uncertainty
        self.kernel_size = args.kernel_size
        self.stride = args.stride
        self.target_size = args.target_size  # <--- newly introduced parameter

        # [NEW] Initialize PCA or UMAP
        self.dim_reduction = args.dim_reduction  # 'pca' or 'umap'
        if self.dim_reduction == 'pca':
            self.pca_components = args.pca_components
            self.pca = PCA(n_components=self.pca_components)
        elif self.dim_reduction == 'umap':
            import umap
            self.umap_n_neighbors = args.umap_n_neighbors
            self.umap_min_dist = args.umap_min_dist
            self.umap_components = args.umap_components
            self.umap = umap.UMAP(n_neighbors=self.umap_n_neighbors, 
                                  min_dist=self.umap_min_dist, 
                                  n_components=self.umap_components)
        
        # [NEW] Specify embedding layers
        self.embedding_layers = args.embedding_layers  # List of layer names

    def get_embedding(self, model, loader, device, args, with_emb=False):
        self.model.eval()
        model = model.to(self.device)

        embeddings_list = []  # To store concatenated embeddings
        self.image_to_embedding_idx = []

        # Initialize pooling
        avg_pool = torch.nn.AvgPool2d(kernel_size=(self.kernel_size, self.kernel_size),
                                      stride=self.stride)

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(tqdm(loader)):
                data = data.to(device)

                if with_emb:
                    embeddings = model(data, with_emb=True)  # List of embeddings from specified layers
                    pooled_embeddings = []
                    for emb in embeddings:
                        while emb.shape[2] * emb.shape[3] > self.target_size:
                            emb = avg_pool(emb)
                        emb = emb.permute(0, 2, 3, 1).reshape(emb.shape[0], -1)
                        pooled_embeddings.append(emb.cpu().numpy())

                    # Concatenate embeddings from multiple layers
                    combined_embedding = np.concatenate(pooled_embeddings, axis=1)
                    embeddings_list.append(combined_embedding)

                # Save indices
                start_idx = batch_idx * data.size(0)
                end_idx = start_idx + data.size(0)
                self.image_to_embedding_idx.extend(range(start_idx, end_idx))

        self.image_to_embedding_idx = np.array(self.image_to_embedding_idx)
        # Stack all embeddings
        combined_embeddings = np.vstack(embeddings_list)

        # Apply dimensionality reduction
        if self.dim_reduction == 'pca':
            combined_embeddings = self.pca.fit_transform(combined_embeddings)
        elif self.dim_reduction == 'umap':
            combined_embeddings = self.umap.fit_transform(combined_embeddings)

        return combined_embeddings, combined_embeddings  # Adjust as needed

    def query(self, n):
        idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
        if self.args.paral:
            train_sampler = DistributedSampler(ActualSequentialSampler(self.train_idx[idxs_unlabeled]))
        else:
            train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])

        data_loader = DataLoader(
            self.dset,
            sampler=train_sampler,
            num_workers=4,
            batch_size=self.batch_size,
            drop_last=False,
            collate_fn=self.custom_collate
        )

        # Getting embeddings
        tgt_emb, tgt_pen_emb = self.get_embedding(self.model, data_loader, self.device, self.args, with_emb=True)
        tgt_emb = tgt_emb  # Already NumPy after reduction

        # Conditionally compute sample_weights
        if self.use_uncertainty:
            # Assuming tgt_emb has logits; adjust if necessary
            tgt_scores = nn.Softmax(dim=1)(torch.from_numpy(tgt_emb) / self.T)
            tgt_scores += 1e-8
            sample_weights = (-(tgt_scores * torch.log(tgt_scores)).sum(1)).cpu().numpy()
        else:
            sample_weights = np.ones(tgt_emb.shape[0])

        km = KMeans(n_clusters=n, random_state=42)
        km.fit(tgt_emb, sample_weight=sample_weights)

        indices = np.arange(tgt_emb.shape[0])
        q_idxs = []
        used_points = set()

        for centroid in km.cluster_centers_:
            distances = np.linalg.norm(tgt_emb[indices] - centroid, axis=1)
            sorted_indices = np.argsort(distances)

            for min_dist_idx in sorted_indices:
                min_index = indices[min_dist_idx]
                if min_index not in used_points:
                    q_idxs.append(min_index)
                    used_points.add(min_index)
                    indices = np.delete(indices, min_dist_idx)
                    break

        image_idxs = self.image_to_embedding_idx[q_idxs]
        image_idxs = list(set(image_idxs))
        return idxs_unlabeled[image_idxs]
