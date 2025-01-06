import os
import sys
import numpy as np
import torch
import time
from datetime import datetime
from torchvision import transforms
from omegaconf import OmegaConf
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from monai.networks.nets import UNet
from clue import CLUESampling  # Update the import path if needed

sys.path.append('../')
from data_utils import MNMv2DataModule
from unet import LightningSegmentationModel
from torch.utils.data import Dataset

# TODO: Add weights and remove later

class MNMv2Subset(Dataset):
    def __init__(
        self,
        input,
        target,
    ):
        self.input = input
        self.target = target

    def __len__(self):
        return self.input.shape[0]
    
    def __getitem__(self, idx):
        return {
            "input": self.input[idx], 
            "target": self.target[idx],
        }

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training or loading a model.")
    parser.add_argument('--train', type=str2bool, nargs='?', const=True, default=True, 
                        help="Whether to train the model")
    parser.add_argument('--num_clusters', type=int, default=5, help="Number of clusters.")
    parser.add_argument('--clue_softmax_t', type=float, default=1.0, help="Temperature.")
    parser.add_argument('--adapt_num_epochs', type=int, default=20, help="Number epochs for finetuning.")
    parser.add_argument('--cluster_type', type=str, default='centroids', 
                        help="This parameter determines whether we will train our model on centroids or on the most confident data close to centroids.")
    parser.add_argument('--checkpoint_path', type=str, 
                        default='/home/chopra/lab-git/MedImSeg-Lab24/checkpoints/mnmv2-15-19_10-12-2024-v1.ckpt', 
                        help="Path to the model checkpoint.")
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help="Device to use for training (e.g., 'cuda:0', 'cuda:1', or 'cpu').")
    parser.add_argument('--paral', type=bool, default=False, 
                        help='Enabling parallelization of the embedding, clustering, and model completion process')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='The threshold removes the images in which the model is most confident')

    # [NEW] Additional parameters for CLUE
    parser.add_argument('--use_uncertainty', type=str2bool, nargs='?', const=True, default=True, 
                        help="Whether to use uncertainty-based sample weights (True) or uniform sample weights (False)")
    parser.add_argument('--kernel_size', type=int, default=3, 
                        help="Kernel size for AvgPool2D in CLUE embedding extraction.")
    parser.add_argument('--stride', type=int, default=2, 
                        help="Stride for AvgPool2D in CLUE embedding extraction.")
    parser.add_argument('--target_size', type=int, default=1024, 
                        help="Target size (spatial area) for pooling in CLUE embedding extraction.")

    # [NEW] Dimensionality Reduction Parameters
    parser.add_argument('--dim_reduction', type=str, choices=['pca', 'umap'], default='pca',
                        help="Dimensionality reduction technique to use.")
    parser.add_argument('--pca_components', type=int, default=100, 
                        help="Number of principal components for PCA.")
    parser.add_argument('--umap_n_neighbors', type=int, default=15, 
                        help="Number of neighbors for UMAP.")
    parser.add_argument('--umap_min_dist', type=float, default=0.1, 
                        help="Minimum distance parameter for UMAP.")
    parser.add_argument('--umap_components', type=int, default=50, 
                        help="Number of components for UMAP.")
    
    # [NEW] Embedding Layers
    parser.add_argument('--embedding_layers', type=str, nargs='+', default=['encoder.layer4', 'decoder.layer4'],
                        help="Names of layers to extract embeddings from.")

    args = parser.parse_args()

    mnmv2_config   = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/mnmv2.yaml')
    unet_config    = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/monai_unet.yaml')
    trainer_config = OmegaConf.load('/home/chopra/lab-git/MedImSeg-Lab24/configs/unet_trainer.yaml')

    # Loop over i to select an increasing number of samples
    for i in [1, 10, 25, 40, 50, 60, 75, 100, 125, 150, 175, 200]:
        # init datamodule
        datamodule = MNMv2DataModule(
            data_dir=mnmv2_config.data_dir,
            vendor_assignment=mnmv2_config.vendor_assignment,
            batch_size=mnmv2_config.batch_size,
            binary_target=mnmv2_config.binary_target,
            non_empty_target=mnmv2_config.non_empty_target,
        )

        cfg = OmegaConf.create({
            'unet_config': unet_config,
            'binary_target': True if unet_config.out_channels == 1 else False,
            'lr': unet_config.lr,
            'patience': unet_config.patience,
            'paral': args.paral,
            'threshold': args.threshold,
            'adapt_num_epochs': args.adapt_num_epochs,
            'cluster_type': args.cluster_type,
            'clue_softmax_t': args.clue_softmax_t,
            'dataset': OmegaConf.to_container(mnmv2_config),
            'batch_size': unet_config.get('batch_size', 32),
            'unet': OmegaConf.to_container(unet_config),
            'trainer': OmegaConf.to_container(trainer_config),

            # [NEW] Pass new params into cfg
            'use_uncertainty': args.use_uncertainty,
            'kernel_size': args.kernel_size,
            'stride': args.stride,
            'target_size': args.target_size,
            'dim_reduction': args.dim_reduction,
            'pca_components': args.pca_components,
            'umap_n_neighbors': args.umap_n_neighbors,
            'umap_min_dist': args.umap_min_dist,
            'umap_components': args.umap_components,
            'embedding_layers': args.embedding_layers,
        })

        if args.train:
            model = LightningSegmentationModel(cfg=cfg)
            
            now = datetime.now()
            filename = 'mnmv2-' + now.strftime("%H-%M_%d-%m-%Y")

            trainer = L.Trainer(
                limit_train_batches=trainer_config.limit_train_batches,
                max_epochs=trainer_config.max_epochs,
                callbacks=[
                    ModelCheckpoint(
                        dirpath=trainer_config.model_checkpoint.dirpath,
                        filename=filename,
                        save_top_k=trainer_config.model_checkpoint.save_top_k, 
                        monitor=trainer_config.model_checkpoint.monitor,
                    )
                ],
                precision='16-mixed',
                devices=[1]  # adapt to your hardware if needed
            )
            trainer.fit(model, datamodule=datamodule)

        else:
            # Handle loading a pre-trained model if not training
            load_as_lightning_module = True
            load_as_pytorch_module = False

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
                    args.checkpoint_path,
                    map_location=torch.device("cpu"),
                    model=unet,
                    binary_target=True if unet_config.out_channels == 1 else False,
                    lr=unet_config.lr,
                    patience=unet_config.patience,
                    cfg=cfg
                )

                trainer = L.Trainer(
                    limit_train_batches=trainer_config.limit_train_batches,
                    max_epochs=args.adapt_num_epochs,
                    callbacks=[
                        ModelCheckpoint(
                            dirpath=trainer_config.model_checkpoint.dirpath,
                            save_top_k=trainer_config.model_checkpoint.save_top_k, 
                            monitor=trainer_config.model_checkpoint.monitor,
                        )
                    ],
                    precision='16-mixed',
                    devices=[1]
                )

            elif load_as_pytorch_module:
                checkpoint = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))
                model_state_dict = checkpoint['state_dict']
                model_state_dict = {k.replace('model.model.', 'model.'): v 
                                    for k, v in model_state_dict.items() 
                                    if k.startswith('model.')}
                model_config = checkpoint['hyper_parameters']['cfgs']

                print(model_config)

                model = UNet(
                    spatial_dims=model_config['unet']['spatial_dims'],
                    in_channels=model_config['unet']['in_channels'],
                    out_channels=model_config['unet']['out_channels'],
                    channels=[model_config['unet']['n_filters_init'] * 2 ** i for i in range(model_config['unet']['depth'])],
                    strides=[2] * (model_config['unet']['depth'] - 1),
                    num_res_units=4
                )

                model.load_state_dict(model_state_dict)
        
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Getting results BEFORE using CLUE
        datamodule.setup(stage='test')
        model.eval()
        test_res = trainer.test(model, datamodule=datamodule)

        # Getting centroids / nearest points to centroids
        test_idx = np.arange(len(datamodule.mnm_test))
        clue_sampler = CLUESampling(
            dset=datamodule.mnm_test,
            train_idx=test_idx,
            model=model,
            device=device,
            args=cfg,
            batch_size=cfg.get('batch_size', 32)  # Pass batch_size explicitly
        )

        # There is no need to set the number of clusters more than the number of images
        if i > len(clue_sampler.dset):
            i = len(clue_sampler.dset)

        start = time.time()
        nearest_idx = clue_sampler.query(n=i)
        end = time.time()
        print("Working Time: ", end - start)

        selected_samples = [datamodule.mnm_test[idx] for idx in nearest_idx]

        # Fine-tuning the model
        datamodule.setup(stage='fit')
        selected_inputs = torch.stack([sample["input"] for sample in selected_samples])
        selected_targets = torch.stack([sample["target"] for sample in selected_samples])

        # Example: fine-tune ONLY on the newly selected data
        combined_data = MNMv2Subset(
            input=selected_inputs,
            target=selected_targets,
        )
        
        # If you want to fine-tune on the entire combined set, you'd do:
        # combined_inputs = torch.cat([datamodule.mnm_train.input, selected_inputs], dim=0)
        # combined_targets = torch.cat([datamodule.mnm_train.target, selected_targets], dim=0)
        # combined_data = MNMv2Subset(
        #     input=combined_inputs,
        #     target=combined_targets,
        # )

        datamodule.mnm_train = combined_data
        new_train_loader = datamodule.train_dataloader()

        model.train()
        trainer.fit(model=model, 
                    train_dataloaders=new_train_loader, 
                    val_dataloaders=datamodule.val_dataloader())

        # Save model after fine-tuning
        if args.cluster_type == 'centroids':
            save_dir = '../pre-trained/finetuned_on_centroids'
        else:
            save_dir = '../pre-trained/finetuned_on_uncert_points'

        os.makedirs(save_dir, exist_ok=True)

        model_save_path = os.path.join(save_dir, f'fituned_model_on_{args.cluster_type}.pth')
        torch.save(model.state_dict(), model_save_path)

        # Getting results AFTER using CLUE
        datamodule.setup(stage='test')
        model = model.to(device)
        model.eval()
        test_perf = trainer.test(model, datamodule=datamodule)[0]

        # Write results to file
        if i == 1:
            with open("/home/chopra/lab-git/MedImSeg-Lab24/results/own_ideas/entropy/results_test_100_32.txt", "w") as f:
                f.write(f"Num_Centroids\tLoss\tDice_Score\tNum_epochs\tCentroid_time\n")    
        
        with open("/home/chopra/lab-git/MedImSeg-Lab24/results/own_ideas/entropy/results_test_100_32.txt", "a") as f:
            f.write(f"{i}\t{test_perf['test_loss']:.4f}\t{test_perf['test_dsc']:.4f}\t{trainer.current_epoch:.4f}\t{end - start:.4f}\n")
