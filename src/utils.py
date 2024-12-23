import torch
from tqdm import tqdm
from monai.losses import DiceCELoss
from torch.utils.data.sampler import Sampler
from torch.utils.data import SubsetRandomSampler
from statistics import mean
from monai.metrics import DiceMetric
import numpy as np
    
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
    
    def custom_collate(self, batch):
        inputs = [item['input'] for item in batch]
        targets = [item['target'] for item in batch]
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        return inputs, targets
    
    # def finetune_model(self, extended_dset, val_dset):
    #     self.model.train()

    #     # DataLoader for the extended dataset (training)
    #     data_loader = torch.utils.data.DataLoader(
    #         extended_dset,
    #         num_workers=4,
    #         batch_size=self.args.unet_config.batch_size,
    #         drop_last=False,
    #         collate_fn=self.custom_collate_train
    #     )
        
    #     # DataLoader for the validation dataset
    #     val_loader = torch.utils.data.DataLoader(
    #         val_dset,
    #         num_workers=4,
    #         batch_size=self.args.unet_config.batch_size,
    #         drop_last=False,
    #         collate_fn=self.custom_collate_val
    #     )

    #     optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    #     criterion = DiceCELoss(
    #         softmax=False if self.args.binary_target else True,
    #         sigmoid=True if self.args.binary_target else False,
    #         to_onehot_y=False if self.args.binary_target else True,
    #     )
    #     dsc_metric = DiceMetric(include_background=False, reduction="none")

    #     for epoch in range(self.args.adapt_num_epochs):
    #         info_str = f"[Finetuning] Epoch: {epoch + 1}"
    #         epoch_loss = 0

    #         # Training loop
    #         for _, (data, target, weight) in enumerate(tqdm(data_loader)):
    #             data, target, weight = data.to(self.device), target.to(self.device), weight.to(self.device)

    #             optimizer.zero_grad()
    #             outputs = self.model(data)
    #             loss = criterion(outputs, target)# * weight).mean()
    #             loss.backward()
    #             optimizer.step()

    #             epoch_loss += loss.item()

    #         info_str += f" Avg Loss: {epoch_loss / len(data_loader):.4f}"
    #         print(info_str)

    #         # Validation loop
    #         self.model.eval()
    #         val_loss = 0
    #         dice_scores = []
    #         with torch.no_grad():
    #             for _, (data, target) in enumerate(val_loader):
    #                 data, target = data.to(self.device), target.to(self.device)
    #                 outputs = self.model(data)
    #                 val_loss += criterion(outputs, target).item()

    #                 # Compute Dice Score
    #                 num_classes = max(outputs.shape[1], 2)
    #                 if num_classes > 2:
    #                     outputs = outputs.argmax(1)
    #                 else:
    #                     outputs = (outputs > 0.5) * 1
    #                 outputs = torch.nn.functional.one_hot(outputs, num_classes=num_classes).moveaxis(-1, 1)
    #                 dice_scores.append(dsc_metric(y_pred=outputs, y=target).nanmean())

    #         avg_val_loss = val_loss / len(val_loader)
    #         avg_dice_score = mean([score.item() for score in dice_scores])
    #         print(f"[Validation] Avg Loss: {avg_val_loss:.4f}, Avg Dice Score: {avg_dice_score:.4f}")

    #         self.model.train()

    #     return self.model

