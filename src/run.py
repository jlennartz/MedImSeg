import sys
import numpy as np
import torch
from omegaconf import OmegaConf
# import lightning as L
# from lightning.pytorch.callbacks import ModelCheckpoint
# from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from monai.networks.nets import UNet
from clue import CLUESampling

sys.path.append('../')
from data_utils import MNMv2DataModule
from unet import LightningSegmentationModel

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
        'adapt_num_epochs': 20,
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

    # Getting the most uncertainty features
    clue_sampler = CLUESampling(dset=datamodule.mnm_train,
                                train_idx=train_idx, 
                                model=model, 
                                device=device, 
                                args=cfg)
    
    #TODO: Change number of clusters in argparse
    nearest_idx = clue_sampler.query(n=2)

    # Getting results BEFORE using CLUE
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()
    start_loss  = model.test_model(test_loader, device)

    # Fine-tuning the model
    idxs_lb = np.zeros(len(train_idx), dtype=bool)
    idxs_lb[nearest_idx] = True
    assert clue_sampler.idxs_lb.sum() == 0, 'Model already updated'
    clue_sampler.update(idxs_lb)

    # Getting results AFTER using CLUE
    new_model = clue_sampler.finetune_model(epochs=1)

    # Testing the model's performance after fine-tuning
    test_perf = new_model.test_model(test_loader, device)
