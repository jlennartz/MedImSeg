from typing import Tuple, Callable
from torch import device, nn, Tensor
from copy import deepcopy
import time

class ModelAdapter(nn.Module):
    """Wrapper for a model and a list of adapters.

    Each adapter is attached to a layer in the model via hooks. The layers
    are defined in the adapters themselves. 
    """
    def __init__(
        self,
        model: nn.Module,
        adapters: nn.ModuleList,
        map_location: device = 'cuda',
        copy: bool = True,
    ):
        super().__init__()
        model = deepcopy(model) if copy else model
        model.to(map_location)
        self.model = model
        self.adapters = adapters
        self.adapter_handles = {}
        self.model.eval()

        self.layer_activations = {}


    def hook_adapters(
        self,
    ) -> None:
        # iterate over all adapters. You can use multiple simultaneously
        for adapter in self.adapters:
            # get the layer name of the layer we want to attach to
            swivel = adapter.swivel
            # get the actual layer from the model
            layer  = self.model.get_submodule(swivel)
            # get the hook function for this layer/adapter
            hook = self._get_hook(adapter, swivel)
            # attach the hook to the layer and save handle to remove hook 
            # if needed 
            self.adapter_handles[
                swivel
            ] = layer.register_forward_pre_hook(hook)


    def _get_hook(
        self,
        adapter: nn.Module,
        swivel: str
    ) -> Callable:
        # registering hooks requires a function that takes the module and the input,
        # nothing more. To access the adapter anyways, we create the hook in a 
        # scope where the adapter is accessible, i.e. in another function.
        def hook_fn(
            module: nn.Module, 
            x: Tuple[Tensor]
        ) -> Tensor:
            # hook signature given by pytorch:
            # hook(module, input) -> None or modified input

            # pass through the adapter
            out = adapter(x[0])
            # save the results from this hook in a dict by layer name
            self.layer_activations[swivel] = adapter.activations.cpu()

            return out
        
        return hook_fn


    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        # hijack the forward pass of your actual model
        return self.model(x)
