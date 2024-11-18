import torch
from tqdm import tqdm
from monai.losses import DiceCELoss
from torch.utils.data.sampler import Sampler
from torch.utils.data import SubsetRandomSampler
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
    
    def custom_collate_fn(self, batch):
        inputs = [item['input'] for item in batch]
        targets = [item['target'] for item in batch]
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        return inputs, targets
    
    def finetune_model(self, epochs):
        self.model.train()

        train_sampler = SubsetRandomSampler(self.train_idx[self.idxs_lb])

        data_loader = torch.utils.data.DataLoader(
            self.dset,
            sampler=train_sampler,
            num_workers=4,
            batch_size=self.args.unet_config.batch_size,
            drop_last=False,
            collate_fn=self.custom_collate_fn
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        criterion = DiceCELoss(
            softmax=False if self.args.binary_target else True,
            sigmoid=True if self.args.binary_target else False,
            to_onehot_y=False if self.args.binary_target else True,
        )

        for epoch in range(self.args.adapt_num_epochs):
            info_str = f"[Finetuning] Epoch: {epoch + 1}"
            epoch_loss = 0

            for _, (data, target) in enumerate(tqdm(data_loader)):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(data)

                loss = criterion(outputs, target)

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()

            info_str += f" Avg Loss: {epoch_loss / len(data_loader):.4f}"
            print(info_str)

            # scheduler.step()

        return self.model
