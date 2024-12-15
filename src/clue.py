import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin
from torch.utils.data import DataLoader, DistributedSampler

from utils import SamplingStrategy, ActualSequentialSampler

class CLUESampling(SamplingStrategy):
    """
    Implements CLUE: Clustering via Uncertainty-weighted Embeddings for segmentation tasks.
    """
    def __init__(self, dset, train_idx, model, device, args, balanced=False):
        super(CLUESampling, self).__init__(dset, train_idx, model, device, args)
        self.dset = dset
        self.train_idx = train_idx
        self.model = model
        self.device = device
        self.args = args
        self.cluster_type = args.cluster_type
        self.T = args.clue_softmax_t

    def get_embedding(self, model, loader, device, args, with_emb=False):
        self.model.eval()
        model = model.to(self.device)
        
        embedding_pen, embedding, emb_dim = None, None, None
        self.pixel_to_image_idx = []

        target_size = 1024
        # avg_pool = torch.nn.AvgPool2d(kernel_size=(3, 3), stride=1)
        avg_pool = torch.nn.AdaptiveAvgPool2d((int(target_size ** 0.5), int(target_size ** 0.5)))
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(tqdm(loader)):
                data = data.to(device)

                if with_emb:
                    e1, e2 = model(data, with_emb=True)

                # Adjust the size of logits to match the embeddings size
                e1 = F.interpolate(e1, size=(e2.shape[2], e2.shape[3]), mode='bilinear', align_corners=False)
                batch_size = e1.shape[0]

                if embedding_pen is None:
                    height, width = e2.shape[2], e2.shape[3]
                    emb_dim = e2.shape[1]
                    num_classes = e1.shape[1]
                    embedding_pen = torch.zeros([len(loader.sampler) * e2.shape[2] * e2.shape[3], emb_dim])
                    embedding = torch.zeros([len(loader.sampler) * e2.shape[2] * e2.shape[3], num_classes])

                # Save image indexes
                for i in range(batch_size):
                    image_idx = batch_idx * batch_size + i
                    self.pixel_to_image_idx.extend([image_idx] * (height * width))
                
                # AvgPooling
                # if e1.shape[2] * e1.shape[3] != target_size:
                #     e1 = avg_pool(e1)
                #     e2 = avg_pool(e2)

                e1 = e1.permute(0, 2, 3, 1).reshape(-1, num_classes)
                e2 = e2.permute(0, 2, 3, 1).reshape(-1, emb_dim)

                start_idx = batch_idx * e2.shape[0]
                end_idx = start_idx + e2.shape[0]

                embedding_pen[start_idx:end_idx, :] = e2.cpu()
                embedding[start_idx:end_idx, :] = e1.cpu()
        
        self.pixel_to_image_idx = np.array(self.pixel_to_image_idx)
        return embedding, embedding_pen
    
    def query(self, n):
        idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
        if self.args.paral:
            train_sampler = DistributedSampler(ActualSequentialSampler(self.train_idx[idxs_unlabeled]))
        else:
            train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
        data_loader = DataLoader(self.dset, 
                                sampler=train_sampler, 
                                num_workers=4,
                                batch_size=self.args.unet_config.batch_size, 
                                drop_last=False,
                                collate_fn=self.custom_collate)
    
        # Obtain embeddings for images
        tgt_emb, tgt_pen_emb = self.get_embedding(self.model, data_loader, self.device, self.args, with_emb=True)
        print(tgt_emb.shape)
        print(tgt_pen_emb.shape)
        # Use the penultimate embeddings (tgt_pen_emb)
        tgt_pen_emb = tgt_pen_emb.cpu().numpy()

        # Calculate uncertainty using entropy for each pixel
        tgt_scores = nn.Softmax(dim=1)(tgt_emb / self.T)
        tgt_scores += 1e-8
        #TODO: Later delete square
        # sample_weights = ((-(tgt_scores * torch.log(tgt_scores)).sum(1))**2).cpu().numpy()
        sample_weights = (-(tgt_scores * torch.log(tgt_scores)).sum(1)).cpu().numpy()
        
        # Set a threshold for uncertainty and filter embeddings
        valid_mask = sample_weights > self.args.threshold

        # Use filtering mask
        filtered_tgt_pen_emb = tgt_pen_emb[valid_mask]
        filtered_sample_weights = sample_weights[valid_mask]
        filtered_pixel_to_image_idx = self.pixel_to_image_idx[valid_mask]

        # Run K-means with uncertainty weights
        km = KMeans(n)
        km.fit(filtered_tgt_pen_emb, sample_weight=filtered_sample_weights)

        dists = euclidean_distances(km.cluster_centers_, filtered_tgt_pen_emb)
        # Find nearest neighbors to inferred centroids
        sort_idxs = dists.argsort(axis=1)
        q_idxs = []
        ax, rem = 0, n
        while rem > 0:
            q_idxs.extend(list(sort_idxs[:, ax][:rem]))
            q_idxs = list(set(q_idxs))
            rem = n-len(q_idxs)
            ax += 1
        
        image_idxs = filtered_pixel_to_image_idx[q_idxs]
        
        image_idxs = list(set(image_idxs))
        
        return idxs_unlabeled[image_idxs]