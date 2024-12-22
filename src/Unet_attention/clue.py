import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin

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
        self.random_state = np.random.RandomState(1234)
        self.T = args.clue_softmax_t
        self.pixel_to_image_idx = []

    def get_embedding(self, model, loader, device, args, with_emb=False):
        """
        Extract embeddings from the model.
        """
        model.eval()
        embedding_pen = None
        embedding = None
        emb_dim = None

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader)):
                # Adjust to handle tuple batches
                if isinstance(batch, tuple):
                    data, _ = batch  # Assume the batch is a tuple of (data, target)
                elif isinstance(batch, dict):
                    data = batch["input"]
                else:
                    raise TypeError(f"Unsupported batch format: {type(batch)}")

                data = data.to(device)

                if with_emb:
                    # Ensure the model returns valid embeddings
                    e1, e2 = model(data, with_emb=True)
                    if e1 is None or e2 is None:
                        raise ValueError("Model returned None for embeddings. Check the model's forward method.")

                # Adjust the size of logits to match the embedding size
                e1 = F.interpolate(e1, size=(e2.shape[2], e2.shape[3]), mode='bilinear', align_corners=False)

                if embedding_pen is None:
                    height, width = e2.shape[2], e2.shape[3]
                    batch_size = e1.shape[0]
                    emb_dim = e2.shape[1]
                    num_classes = e1.shape[1]
                    embedding_pen = torch.zeros([len(loader.dataset) * height * width, emb_dim])
                    embedding = torch.zeros([len(loader.dataset) * height * width, num_classes])

                e1 = e1.permute(0, 2, 3, 1).reshape(-1, num_classes)
                e2 = e2.permute(0, 2, 3, 1).reshape(-1, emb_dim)

                # Save pixel-to-image mapping
                for i in range(batch_size):
                    image_idx = batch_idx * batch_size + i
                    self.pixel_to_image_idx.extend([image_idx] * (height * width))

                # Fill tensors
                start_idx = batch_idx * e2.shape[0]
                end_idx = start_idx + e2.shape[0]
                embedding_pen[start_idx:end_idx, :] = e2.cpu()
                embedding[start_idx:end_idx, :] = e1.cpu()

        return embedding, embedding_pen, height, width

    def query(self, n):
        """
        Query the model to select the most informative samples.
        """
        idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
        train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
        data_loader = torch.utils.data.DataLoader(
            self.dset,
            sampler=train_sampler,
            num_workers=4,
            batch_size=self.args.unet_config.batch_size,
            drop_last=False,
            collate_fn=self.custom_collate_fn
        )
        self.model.eval()

        # Obtain embeddings for pixels
        tgt_emb, tgt_pen_emb, height, width = self.get_embedding(self.model, data_loader, self.device, self.args, with_emb=True)

        # Use the penultimate embeddings (tgt_pen_emb)
        tgt_pen_emb = tgt_pen_emb.cpu().numpy()

        # Calculate uncertainty using entropy for each pixel
        tgt_scores = nn.Softmax(dim=1)(tgt_emb / self.T)
        tgt_scores += 1e-8  # Avoid log(0)
        sample_weights = -(tgt_scores * torch.log(tgt_scores)).sum(1).cpu().numpy()

        # Run K-means with uncertainty weights
        km = KMeans(n)
        km.fit(tgt_pen_emb, sample_weight=sample_weights)

        # Determine query indices based on cluster type
        if self.cluster_type == 'centroids':
            q_idxs = pairwise_distances_argmin(km.cluster_centers_, tgt_pen_emb)
        else:  # Handle 'uncert_points' or other cluster types
            dists = euclidean_distances(km.cluster_centers_, tgt_pen_emb)
            sort_idxs = dists.argsort(axis=1)
            q_idxs = []
            ax, rem = 0, n
            while rem > 0:
                q_idxs.extend(list(sort_idxs[:, ax][:rem]))
                q_idxs = list(set(q_idxs))
                rem = n - len(q_idxs)
                ax += 1

        # Map pixel indices to image indices
        pixel_to_image_idx = np.array(self.pixel_to_image_idx)
        image_idxs = pixel_to_image_idx[q_idxs]
        image_idxs = list(set(image_idxs))

        return idxs_unlabeled[image_idxs]
