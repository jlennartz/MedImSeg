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

sys.path.append('../')
from data_utils import MNMv2DataModule
from unet import LightningSegmentationModel
import torch.nn.functional as F


class CLUESampling:
    """
    Implements CLUE: Clustering via Uncertainty-weighted Embeddings for segmentation tasks.
    """
    def __init__(self, dset, model, device, args, balanced=False):
        self.dset = dset
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
        batch_sz = args.batch_size
        num_samples = len(self.dset)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader)):
                data = batch['data'].to(device)

                if with_emb:
                    e1, e2 = model(data, with_emb=True)
                    height, width = e2.shape[2], e2.shape[3]
                # else:
                #     e1 = model(data, with_emb=False)

                # Adjust the size of logits to match the embeddings size
                e1 = F.interpolate(e1, size=(height, width), mode='bilinear', align_corners=False)

                if embedding_pen is None:
                    emb_dim = e2.shape[1]
                    num_classes = e1.shape[1]
                    embedding_pen = torch.zeros((num_samples * height * width, emb_dim), device='cpu')
                    embedding = torch.zeros((num_samples * height * width, num_classes), device='cpu')

                # Transform logits and embeddings for each pixel
                e1 = e1.permute(0, 2, 3, 1).reshape(-1, num_classes)
                e2 = e2.permute(0, 2, 3, 1).reshape(-1, emb_dim)

                # Calculate current indices
                start_idx = batch_idx * batch_sz * height * width
                end_idx = start_idx + min(batch_sz * height * width, e2.shape[0])

                # Fill tensors
                embedding[start_idx:end_idx, :] = e1.cpu()
                embedding_pen[start_idx:end_idx, :] = e2.cpu()

                # if batch_idx > 50:
                #     break
        
        return embedding, embedding_pen
    
    def query(self, n, data_loader):
        self.model.eval()

        # Obtain embeddings for pixels
        tgt_emb, tgt_pen_emb = self.get_embedding(self.model, data_loader, self.device, self.args, with_emb=True)

        # Use the penultimate embeddings (tgt_pen_emb)
        tgt_pen_emb = tgt_pen_emb.cpu().numpy()

        # Calculate uncertainty using entropy for each pixel
        tgt_scores = nn.Softmax(dim=1)(tgt_emb / self.T)
        tgt_scores += 1e-8
        sample_weights = -(tgt_scores * torch.log(tgt_scores)).sum(1).cpu().numpy()

        # Run K-means with uncertainty weights
        km = KMeans(n)
        km.fit(tgt_pen_emb, sample_weight=sample_weights)

        # Return the centroid embeddings
        return km.cluster_centers_

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
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    clue_sampler = CLUESampling(dset=datamodule.mnm_train, 
                                model=model, 
                                device=device, 
                                args=unet_config, )
                                            #cache_path='../../MedImSeg-Lab24/checkpoints/emb_and_weights.pkl')
    # Change number of clusters
    centroids = clue_sampler.query(n=2, data_loader=train_loader)

    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()
    start_loss  = model.test_model(test_loader, device)
    # out_str = '{} | Test performance on {}->{}: Round 0 (B=0): {:.2f}'.format(method, source, target, start_perf)

    # Step of fine-tuning the model using the centroids
    model.finetune_model_on_centroids(centroids, train_loader, model)

    # Testing the model's performance after fine-tuning
    test_perf = model.test_model(test_loader, device)
    # out_str += '\t Round 1 (B={}): {:.2f}'.format(len(cluster_centers), test_perf)

    # Output the results and print performance before and after training
    # print(start_loss['dsc'], test_perf['dsc'])
