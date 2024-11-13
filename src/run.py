import os
import glob
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

import sys, string, random
from datetime import datetime
from omegaconf import OmegaConf
import wandb
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from monai.networks.nets import UNet
from monai.losses import DiceCELoss

sys.path.append('../')
from data_utils import MNMv2DataModule
from unet import LightningSegmentationModel
import torch.nn.functional as F
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

T_co = TypeVar('T_co', covariant=True)

class Sampler(Generic[T_co]):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices or lists of indices (batches) of dataset elements,
    and may provide a :meth:`__len__` method that returns the length of the returned iterators.

    Args:
        data_source (Dataset): This argument is not used and will be removed in 2.2.0.
            You may still have custom implementation that utilizes it.

    Example:
        >>> # xdoctest: +SKIP
        >>> class AccedingSequenceLengthSampler(Sampler[int]):
        >>>     def __init__(self, data: List[str]) -> None:
        >>>         self.data = data
        >>>
        >>>     def __len__(self) -> int:
        >>>         return len(self.data)
        >>>
        >>>     def __iter__(self) -> Iterator[int]:
        >>>         sizes = torch.tensor([len(x) for x in self.data])
        >>>         yield from torch.argsort(sizes).tolist()
        >>>
        >>> class AccedingSequenceLengthBatchSampler(Sampler[List[int]]):
        >>>     def __init__(self, data: List[str], batch_size: int) -> None:
        >>>         self.data = data
        >>>         self.batch_size = batch_size
        >>>
        >>>     def __len__(self) -> int:
        >>>         return (len(self.data) + self.batch_size - 1) // self.batch_size
        >>>
        >>>     def __iter__(self) -> Iterator[List[int]]:
        >>>         sizes = torch.tensor([len(x) for x in self.data])
        >>>         for batch in torch.chunk(torch.argsort(sizes), len(self)):
        >>>             yield batch.tolist()

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source: Optional[Sized] = None) -> None:
        if data_source is not None:
            import warnings

            warnings.warn("`data_source` argument is not used and will be removed in 2.2.0."
                          "You may still have custom implementation that utilizes it.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

    # NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    #
    # Many times we have an abstract class representing a collection/iterable of
    # data, e.g., `torch.utils.data.Sampler`, with its subclasses optionally
    # implementing a `__len__` method. In such cases, we must make sure to not
    # provide a default implementation, because both straightforward default
    # implementations have their issues:
    #
    #   + `return NotImplemented`:
    #     Calling `len(subclass_instance)` raises:
    #       TypeError: 'NotImplementedType' object cannot be interpreted as an integer
    #
    #   + `raise NotImplementedError`:
    #     This prevents triggering some fallback behavior. E.g., the built-in
    #     `list(X)` tries to call `len(X)` first, and executes a different code
    #     path if the method is not found or `NotImplemented` is returned, while
    #     raising a `NotImplementedError` will propagate and make the call fail
    #     where it could have used `__iter__` to complete the call.
    #
    # Thus, the only two sensible things to do are
    #
    #   + **not** provide a default `__len__`.
    #
    #   + raise a `TypeError` instead, which is what Python uses when users call
    #     a method that is not defined on an object.
    #     (@ssnl verifies that this works on at least Python 3.7.)

class SubsetRandomSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """

    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for i in torch.randperm(len(self.indices), generator=self.generator):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)
    
class ActualSequentialSampler(Sampler):
	r"""Samples elements sequentially, always in the same order.

	Arguments:
		data_source (Dataset): dataset to sample from
	"""

	def __init__(self, data_source):
		self.data_source = data_source

	def __iter__(self):
		return iter(self.data_source)

	def __len__(self):
		return len(self.data_source)

class SamplingStrategy:
    """ 
    Sampling Strategy wrapper class
    """
    def __init__(self, dset, train_idx, model, device, args):
        self.dset = dset
        self.train_idx = np.array(train_idx)
        self.model = model
        self.device = device
        self.args = args
        self.idxs_lb = np.zeros(len(self.train_idx), dtype=bool)

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb
    
    def query(self, n):
        pass
    
    def custom_collate_fn(self, batch):
        inputs = [item['input'] for item in batch]
        targets = [item['target'] for item in batch]
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        return inputs, targets
    
    def finetune_model(self):
        self.model.train()

        train_sampler = SubsetRandomSampler(self.train_idx[self.idxs_lb])
        data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler, num_workers=4, \
												 	 batch_size=self.args.batch_size, drop_last=False, collate_fn=self.custom_collate_fn)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        loss = DiceCELoss(
            softmax=False if cfg.binary_target else True,
            sigmoid=True if cfg.binary_target else False,
            to_onehot_y=False if cfg.binary_target else True,
        )
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            
            optimizer.zero_grad()
            data, target = data.to(self.device), target.to(self.device)
            outputs = self.model(data)
            loss = loss(outputs, target)
    
            loss.backward()
            optimizer.step()
        
        return self.model

 
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
        self.random_state = np.random.RandomState(1234)
        self.T = args.clue_softmax_t

    def get_embedding(self, model, loader, device, args, with_emb=False):
        model.eval()
        embedding_pen = None
        embedding = None
        emb_dim = None
        self.pixel_to_image_idx = []
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(tqdm(loader)):
                data = data.to(device)

                if with_emb:
                    e1, e2 = model(data, with_emb=True)
                    
                    # Use AvgPool for dimension reduction
                    # e2 = F.avg_pool2d(e2, kernel_size=3, stride=2)
                # else:
                #     e1 = model(data, with_emb=False)

                # Adjust the size of logits to match the embeddings size
                e1 = F.interpolate(e1, size=(e2.shape[2], e2.shape[3]), mode='bilinear', align_corners=False)
                
                if embedding_pen is None:
                    height, width = e2.shape[2], e2.shape[3]
                    batch_size = e1.shape[0]
                    emb_dim = e2.shape[1]
                    num_classes = e1.shape[1]
                    embedding_pen = torch.zeros([len(loader.sampler) * e2.shape[2] * e2.shape[3], emb_dim])
                    embedding = torch.zeros([len(loader.sampler) * e2.shape[2] * e2.shape[3], num_classes])

                e1 = e1.permute(0, 2, 3, 1).reshape(-1, num_classes)
                e2 = e2.permute(0, 2, 3, 1).reshape(-1, emb_dim)

                # Save image indexes
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
        idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
        train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
        data_loader = torch.utils.data.DataLoader(self.dset, 
                                                sampler=train_sampler, 
                                                num_workers=4,
												batch_size=self.args.batch_size, 
                                                drop_last=False,
                                                collate_fn=self.custom_collate_fn)
        self.model.eval()


        # Obtain embeddings for pixels
        tgt_emb, tgt_pen_emb, height, width = self.get_embedding(self.model, data_loader, self.device, self.args, with_emb=True)

        # Use the penultimate embeddings (tgt_pen_emb)
        tgt_pen_emb = tgt_pen_emb.cpu().numpy()

        # Calculate uncertainty using entropy for each pixel
        tgt_scores = nn.Softmax(dim=1)(tgt_emb / self.T)
        tgt_scores += 1e-8
        sample_weights = -(tgt_scores * torch.log(tgt_scores)).sum(1).cpu().numpy()

        # Run K-means with uncertainty weights
        km = KMeans(n)
        km.fit(tgt_pen_emb, sample_weight=sample_weights)

        # Find nearest neighbors to inferred centroids
        dists = euclidean_distances(km.cluster_centers_, tgt_pen_emb)
        sort_idxs = dists.argsort(axis=1)
        q_idxs = []
        ax, rem = 0, n
        while rem > 0:
            q_idxs.extend(list(sort_idxs[:, ax][:rem]))
            q_idxs = list(set(q_idxs))
            rem = n-len(q_idxs)
            ax += 1
        
        # q_idxs = np.array(q_idxs)
        # assert np.all(q_idxs % (height * width) == 0), "Некоторые индексы пикселей не делятся на (height * width)"
        # Convert pixel indices to image indices
        # image_idxs = q_idxs // (height * width)

        pixel_to_image_idx = np.array(self.pixel_to_image_idx)
        image_idxs = pixel_to_image_idx[q_idxs]
        image_idxs = list(set(image_idxs))
        
        return idxs_unlabeled[image_idxs]

if __name__ == '__main__':
    mnmv2_config   = OmegaConf.load('../../MedImSeg-Lab24/configs/mnmv2.yaml')
    unet_config    = OmegaConf.load('../../MedImSeg-Lab24/configs/monai_unet.yaml')
    trainer_config = OmegaConf.load('../../MedImSeg-Lab24/configs/unet_trainer.yaml')

    # init datamodule
    datamodule = MNMv2DataModule(
        data_dir=mnmv2_config.data_dir,
        vendor_assignment=mnmv2_config.vendor_assignment,
        batch_size=mnmv2_config.batch_size,
        binary_target=mnmv2_config.binary_target,
        non_empty_target=mnmv2_config.non_empty_target,
    )

    datamodule.setup(stage='fit')

    # init model
    cfg = OmegaConf.create({
        'unet_config': unet_config,
        'binary_target': True if unet_config.out_channels == 1 else False,
        'lr': unet_config.lr,
        'patience': unet_config.patience,
        'lambda_centroids': 0.6,
        'dataset': OmegaConf.to_container(mnmv2_config),
        'unet': OmegaConf.to_container(unet_config),
        'trainer': OmegaConf.to_container(trainer_config),
    })

    # TODO: Add argument for training 

    # model = LightningSegmentationModel(cfg=cfg)

    # # infered variable
    # patience = unet_config.patience * 2

    # now = datetime.now()
    # filename = 'mnmv2-' + now.strftime("%H-%M_%d-%m-%Y")

    # # init trainer
    # if trainer_config.logging:
    #     wandb.finish()
    #     logger = WandbLogger(
    #         project="lightning", 
    #         log_model=True, 
    #         name=filename
    #     )
    # else:
    #     logger = None

    # # trainer
    # trainer = L.Trainer(
    #     limit_train_batches=trainer_config.limit_train_batches,
    #     max_epochs=trainer_config.max_epochs,
    #     logger=logger,
    #     callbacks=[
    #         EarlyStopping(
    #             monitor=trainer_config.early_stopping.monitor, 
    #             mode=trainer_config.early_stopping.mode, 
    #             patience=patience
    #         ),
    #         ModelCheckpoint(
    #             dirpath=trainer_config.model_checkpoint.dirpath,
    #             filename=filename,
    #             save_top_k=trainer_config.model_checkpoint.save_top_k, 
    #             monitor=trainer_config.model_checkpoint.monitor,
    #         )
    #     ],
    #     precision='16-mixed',
    #     gradient_clip_val=0.5,
    #     devices=[0]
    # )

    # trainer.fit(model, datamodule=datamodule)

    checkpoint_path = '../../MedImSeg-Lab24/pre-trained/trained_UNets/mnmv2-10-12_06-11-2024.ckpt'

    load_as_lightning_module = True#False
    load_as_pytorch_module = False#True

    if load_as_lightning_module:
        unet_config    = OmegaConf.load('../../MedImSeg-Lab24/configs/monai_unet.yaml')
        unet = UNet(
            spatial_dims=unet_config.spatial_dims,
            in_channels=unet_config.in_channels,
            out_channels=unet_config.out_channels,
            channels=[unet_config.n_filters_init * 2 ** i for i in range(unet_config.depth)],
            strides=[2] * (unet_config.depth - 1),
            num_res_units=4
        )
        
        model = LightningSegmentationModel.load_from_checkpoint(
            checkpoint_path,
            map_location=torch.device("cpu"),
            model=unet,
            binary_target=True if unet_config.out_channels == 1 else False,
            lr=unet_config.lr,
            patience=unet_config.patience,
            cfg=cfg
        )

    elif load_as_pytorch_module:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        model_state_dict = checkpoint['state_dict']
        model_state_dict = {k.replace('model.model.', 'model.'): v for k, v in model_state_dict.items() if k.startswith('model.')}
        model_config = checkpoint['hyper_parameters']['cfgs']

        print(model_config)

        unet = UNet(
            spatial_dims=model_config['unet']['spatial_dims'],
            in_channels=model_config['unet']['in_channels'],
            out_channels=model_config['unet']['out_channels'],
            channels=[model_config['unet']['n_filters_init'] * 2 ** i for i in range(model_config['unet']['depth'])],
            strides=[2] * (model_config['unet']['depth'] - 1),
            num_res_units=4
        )

        unet.load_state_dict(model_state_dict)
    
    train_loader = datamodule.train_dataloader()
    train_idx = np.arange(len(datamodule.mnm_train))
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    clue_sampler = CLUESampling(dset=datamodule.mnm_train,
                                train_idx=train_idx, 
                                model=model, 
                                device=device, 
                                args=unet_config)
    # Change number of clusters
    nearest_idx = clue_sampler.query(n=6)

    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()
    start_loss  = model.test_model(test_loader, device)

    # Fine-tuning the model
    idxs_lb = np.zeros(len(train_idx), dtype=bool)
    idxs_lb[nearest_idx] = True
    assert clue_sampler.idxs_lb.sum() == 0, 'Model already updated'
    clue_sampler.update(idxs_lb)

    new_model = clue_sampler.finetune_model()

    # Testing the model's performance after fine-tuning
    test_perf = new_model.test_model(test_loader, device)
