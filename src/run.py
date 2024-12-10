import os
import sys
import numpy as np
import torch
from datetime import datetime
from omegaconf import OmegaConf
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from monai.networks.nets import UNet
from clue import CLUESampling

sys.path.append('../')
from data_utils import MNMv2DataModule
from unet import LightningSegmentationModel
from torch.utils.data import Dataset

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
            "target": self.target[idx]
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
    parser.add_argument('--train', type=str2bool, nargs='?', const=True, default=True, help="Whether to train the model")
    parser.add_argument('--n', type=int, default=5, help="Number of clusters.")
    parser.add_argument('--clue_softmax_t', type=float, default=1.0, help="Temperature.")
    parser.add_argument('--adapt_num_epochs', type=int, default=20, help="Number epochs for finetuning.")
    parser.add_argument('--cluster_type', type=str, default='centroids', help="This parameter determines whether we will train our model on centroids or on the most confident data close to centroids.")
    parser.add_argument('--checkpoint_path', type=str, default='../../MedImSeg-Lab24/pre-trained/trained_UNets/mnmv2-10-12_06-11-2024.ckpt', 
                        help="Path to the model checkpoint.")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to use for training (e.g., 'cuda:0', 'cuda:1', or 'cpu').")
    args = parser.parse_args()

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

    cfg = OmegaConf.create({
        'unet_config': unet_config,
        'binary_target': True if unet_config.out_channels == 1 else False,
        'lr': unet_config.lr,
        'patience': unet_config.patience,
        'adapt_num_epochs': args.adapt_num_epochs,
        'cluster_type': args.cluster_type,
        'clue_softmax_t': args.clue_softmax_t,
        'dataset': OmegaConf.to_container(mnmv2_config),
        'unet': OmegaConf.to_container(unet_config),
        'trainer': OmegaConf.to_container(trainer_config),
    })

    if args.train:
        model = LightningSegmentationModel(cfg=cfg)
        
        now = datetime.now()
        filename = 'mnmv2-' + now.strftime("%H-%M_%d-%m-%Y")

        trainer = L.Trainer(
            limit_train_batches=trainer_config.limit_train_batches,
            max_epochs=trainer_config.max_epochs,
            callbacks=[
                EarlyStopping(
                    monitor=trainer_config.early_stopping.monitor, 
                    mode=trainer_config.early_stopping.mode, 
                    patience=unet_config.patience * 2
                ),
                ModelCheckpoint(
                    dirpath=trainer_config.model_checkpoint.dirpath,
                    filename=filename,
                    save_top_k=trainer_config.model_checkpoint.save_top_k, 
                    monitor=trainer_config.model_checkpoint.monitor,
                )
            ],
            precision='16-mixed',
            gradient_clip_val=0.5,
            devices=[0]
        )

        trainer.fit(model, datamodule=datamodule)

    else:
        #TODO: Add argsparse
        load_as_lightning_module = True #False
        load_as_pytorch_module = False #True

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

        elif load_as_pytorch_module:
            checkpoint = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))
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
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Getting the most uncertainty features
    datamodule.setup(stage='fit')
    val_idx = np.arange(len(datamodule.mnm_val))
    clue_sampler = CLUESampling(dset=datamodule.mnm_val,
                                train_idx=val_idx, 
                                model=model, 
                                device=device, 
                                args=cfg)
    # Getting centroids / nearest points to centroids
    nearest_idx = clue_sampler.query(n=args.n)
    selected_samples = [datamodule.mnm_val[i] for i in nearest_idx]

    # Getting results BEFORE using CLUE
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()
    start_loss  = model.test_model(test_loader, device)

    # Fine-tuning the model
    # Extend train data by test samples with the highest uncertainty
    datamodule.setup(stage='fit')

    selected_inputs = torch.stack([sample["input"] for sample in selected_samples])
    selected_targets = torch.stack([sample["target"] for sample in selected_samples])

    # Combining input data and labels
    combined_inputs = torch.cat([datamodule.mnm_train.input, selected_inputs], dim=0)
    combined_targets = torch.cat([datamodule.mnm_train.target, selected_targets], dim=0)

    datamodule.mnm_train = MNMv2Subset(
        input=combined_inputs,
        target=combined_targets
    )
    new_model = clue_sampler.finetune_model(datamodule.mnm_train, datamodule.mnm_val)

    if args.cluster_type == 'centroids':
        save_dir = '../pre-trained/finetuned_on_centroids'
    else:
        save_dir = '../pre-trained/finetuned_on_uncert_points'

    os.makedirs(save_dir, exist_ok=True)

    model_save_path = os.path.join(save_dir, f'fituned_model_on_{args.cluster_type}.pth')
    torch.save(model.state_dict(), model_save_path)

    # Getting results AFTER using CLUE
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()
    test_perf = new_model.test_model(test_loader, device)